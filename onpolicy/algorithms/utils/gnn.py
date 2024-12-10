import numpy as np
from scipy import sparse
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, to_dense_batch

import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch.jit as jit

# Assuming you have init and get_clones in util.py
from .util import init, get_clones

"""GNN modules"""

class EmbedConv(MessagePassing):
    def __init__(self, 
                input_dim:int, 
                num_embeddings:int, 
                embedding_size:int, 
                hidden_size:int,
                layer_N:int,
                use_orthogonal:bool,
                use_ReLU:bool,
                use_layerNorm:bool,
                add_self_loop:bool,
                edge_dim:int=0):
        """
            EmbedConv Layer which takes in node features, node_type (entity type) 
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and 
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int): 
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use ReLU for each layer
            use_layerNorm (bool): 
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional): 
                Edge feature dimension, If zero then edge features are not 
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(aggr='add')
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity()
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        
        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        
        # Define the first linear layer
        self.lin1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)
        
        # Initialize the hidden layers
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)
        
        # Apply initialization
        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x:Union[Tensor, OptPairTensor], edge_index:Adj,
                edge_attr:OptTensor=None):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if isinstance(x, Tensor):
            x : OptPairTensor = (x, x)
        
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j:Tensor, edge_attr:OptTensor):
        """
            The node_obs obtained from the environment
            is actually [node_features, node_num, entity_type]
            x_i' = AGG([x_j, EMB(ent_j), e_ij] : j \in \mathcal{N}(i))
        """
        node_feat_j = x_j[:,:-1]
        entity_type_j = x_j[:,-1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        # Apply the first linear layer
        x = self.lin1(node_feat)
        x = self.active_func(x)
        x = self.layer_norm(x)
        
        # Apply the hidden layers
        for layer in self.layers:
            x = layer(x)
        
        return x

class GATNet(nn.Module):
    def __init__(self,
                input_dim:int,
                num_embeddings:int,
                embedding_size:int,
                hidden_size:int,
                num_heads:int,
                concat_heads:bool,
                layer_N:int,
                use_ReLU:bool,
                graph_aggr:str,
                global_aggr_type:str,
                embed_hidden_size:int,
                embed_layer_N:int,
                embed_use_orthogonal:bool,
                embed_use_ReLU:bool,
                embed_use_layerNorm:bool,
                embed_add_self_loop:bool,
                max_edge_dist:float,
                edge_dim:int=1,
                ):
        """
            Module for GATConv Net:
            • This will process the adjacency weight matrix, construct the binary 
                adjacency matrix according to `max_edge_dist` parameter, assign 
                edge weights as the weights in the adjacency weight matrix.
            • After this, the batch data is converted to a PyTorch Geometric 
                compatible dataloader.
            • Then the batch is passed through the graph neural network.
            • The node feature output is then either:
                • Aggregated across the graph to get graph encoded data.
                • Pull node specific `message_passed` hidden feature as output.

        Args:
            input_dim (int):
                Node feature dimension
                NOTE: a reduction of `input_dim` by 1 will be carried out 
                internally because `node_obs` = [node_feat, entity_type]
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the attention layers
            num_heads (int):
                Number of heads in the attention layer
            concat_heads (bool):
                Whether to concatenate the heads in the attention layer or 
                average them
            layer_N (int):
                Number of attention layers for aggregation
            use_ReLU (bool):
                Whether to use ReLU for each layer
            graph_aggr (str):
                Whether we want to pull node specific features from the output or
                perform global_pool on all nodes. 
                Choices: ['global', 'node']
            global_aggr_type (str):
                The type of aggregation to perform if `graph_aggr` is `global`
                Choices: ['mean', 'max', 'add']
            embed_hidden_size (int):
                Hidden layer size of the linear layers in `EmbedConv`
            embed_layer_N (int):
                Number of linear layers for aggregation in `EmbedConv`
            embed_use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer in `EmbedConv`
            embed_use_ReLU (bool):
                Whether to use ReLU for each layer in `EmbedConv`
            embed_use_layerNorm (bool):
                Whether to use layerNorm for each layer in `EmbedConv`
            embed_add_self_loop (bool):
                Whether to add self loops in the graph in `EmbedConv`
            max_edge_dist (float):
                The maximum edge distance to consider while constructing the graph
            edge_dim (int, optional): 
                Edge feature dimension, If zero then edge features are not 
                considered in `EmbedConv`. Defaults to 1.
        """
        super(GATNet, self).__init__()
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        # NOTE: reducing dimension of input by 1 because 
        # node_obs = [node_feat, entity_type]
        self.embed_layer = EmbedConv(input_dim=input_dim-1, 
                            num_embeddings=num_embeddings, 
                            embedding_size=embedding_size, 
                            hidden_size=embed_hidden_size,
                            layer_N=embed_layer_N, 
                            use_orthogonal=embed_use_orthogonal, 
                            use_ReLU=embed_use_ReLU, 
                            use_layerNorm=embed_use_layerNorm, 
                            add_self_loop=embed_add_self_loop,
                            edge_dim=edge_dim)
        # First GATConv layer
        self.gnn1 = GATConv(in_channels=embed_hidden_size, 
                            out_channels=hidden_size,
                            heads=num_heads, 
                            concat=concat_heads,
                            dropout=0.0, 
                            edge_dim=edge_dim,
                            add_self_loops=False,
                            bias=True)
        self.gnn2 = nn.ModuleList()
        for _ in range(layer_N):
            in_channels = hidden_size * num_heads if concat_heads else hidden_size
            self.gnn2.append(GATConv(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       heads=self.num_heads,
                                       concat=self.concat_heads,
                                       dropout=0.0,
                                       edge_dim=self.edge_dim,
                                       add_self_loops=False,
                                       bias=True))
        self.activation = nn.ReLU() if use_ReLU else nn.Tanh()
    
    def forward(self, batch):
        """
            node_obs: Tensor shape:(batch_size, num_nodes, node_obs_dim)
                Node features in the graph formed wrt agent_i
            adj: Tensor shape:(batch_size, num_nodes, num_nodes)
                Adjacency Matrix for the graph formed wrt agent_i
                NOTE: Right now the adjacency matrix is the distance
                magnitude between all entities so will have to post-process
                this to obtain the edge_index and edge_attr
            agent_id: Tensor shape:(batch_size) or (batch_size, k)
                Node number for agent_i in the graph. This will be used to
                pull out the aggregated features from that node
        """

        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.embed_layer(x, edge_index, edge_attr)
        # Forward pass through first GATConv
        x = self.activation(self.gnn1(x, edge_index, edge_attr))
        for gnn in self.gnn2:
            x = self.activation(gnn(x, edge_index, edge_attr))
        if self.graph_aggr == 'node':
            return x
        elif self.graph_aggr == 'global':
            if self.global_aggr_type == 'mean':
                return global_mean_pool(x, batch.batch)
            elif self.global_aggr_type == 'max':
                return global_max_pool(x, batch.batch)
            elif self.global_aggr_type == 'add':
                return global_add_pool(x, batch.batch)
        raise ValueError(f"Invalid graph_aggr: {self.graph_aggr}")

    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        """
        Process adjacency matrix to filter far away nodes
        and then obtain the edge_index and edge_weight
        `adj` is of shape (batch_size, num_nodes, num_nodes)
        OR (num_nodes, num_nodes)
        """
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        # filter far away nodes and connection to itself
        connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
        adj = adj * connect_mask

        if adj.dim() == 3:
            # Case: (batch_size, num_nodes, num_nodes)
            batch_size, num_nodes, _ = adj.shape
            edge_index = adj.nonzero(as_tuple=False)
            edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
            # Adjust indices for batched graph
            batch = edge_index[:, 0] * num_nodes
            edge_index = torch.stack([batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0)
        else:
            # Case: (num_nodes, num_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]]

        # Ensure edge_attr is 2D
        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr

        return edge_index, edge_attr

    def gatherNodeFeats(self, x:Tensor, idx:Tensor):
        """
            The output obtained from the network is of shape 
            [batch_size, num_nodes, out_channels]. If we want to 
            pull the features according to particular nodes in the 
            graph as determined by the `idx`, use this
            Refer below link for more info on `gather()` method for 3D tensors
            https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

            Args:
                x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
                idx (Tensor): Tensor of shape (batch_size) or (batch_size, k) 
                    indicating the indices of nodes to pull from the graph

            Returns:
                Tensor: Tensor of shape (batch_size, out_channels) which just 
                    contains the features from the node of interest
        """
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1) # (batch_size, 1)
            assert idx_tmp.shape == (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats)  # (batch_size, out_channels)
            idx_tmp = idx_tmp.unsqueeze(1)      # (batch_size, 1, out_channels)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)  # (batch_size, out_channels)
            out.append(gathered_node)    
        out = torch.cat(out, dim=1) # (batch_size, out_channels*k)
        return out

    def graphAggr(self, x:Tensor):
        """
            Aggregate the graph node features by performing global pool

            Args:
                x (Tensor): Tensor of shape [batch_size, num_nodes, num_feats]
                aggr (str): Aggregation method for performing the global pool

            Raises:
                ValueError: If `aggr` is not in ['mean', 'max', 'add']

            Returns:
                Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if self.global_aggr_type =='mean':
            return x.mean(dim=1)
        elif self.global_aggr_type == 'max':
            max_feats, idx = x.max(dim=1)
            return max_feats
        elif self.global_aggr_type == 'add':
            return x.sum(dim=1)
        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")

class GNNBase(nn.Module):
    """
        A Wrapper for constructing the Base graph neural network.
        This uses GATConv from PyTorch Geometric
        and embedding layers for entity types
        Params:
        args: (argparse.Namespace)
            Should contain the following arguments
            num_embeddings: (int)
                Number of entity types in the env to have different embeddings 
                for each entity type
            embedding_size: (int)
                Embedding layer output size for each entity category
            embed_hidden_size: (int)
                Hidden layer dimension after the embedding layer
            embed_layer_N: (int)
                Number of hidden linear layers after the embedding layer")
            embed_use_ReLU: (bool)
                Whether to use ReLU in the linear layers after the embedding layer
            embed_add_self_loop: (bool)
                Whether to add self loops in adjacency matrix
            gnn_hidden_size: (int)
                Hidden layer dimension in the GNN
            gnn_num_heads: (int)
                Number of heads in the GATConv layer (GNN)
            gnn_concat_heads: (bool)
                Whether to concatenate the head output or average
            gnn_layer_N: (int)
                Number of GNN conv layers
            gnn_use_ReLU: (bool)
                Whether to use ReLU in GNN conv layers
            max_edge_dist: (float)
                Maximum distance above which edges cannot be connected between 
                the entities
            graph_feat_type: (str)
                Whether to use 'global' node/edge feats or 'relative'
                choices=['global', 'relative']
        node_obs_shape: (Union[Tuple, List])
            The node observation shape. Example: (18,)
        edge_dim: (int)
            Dimensionality of edge attributes 
    """
    def __init__(self, args:argparse.Namespace, 
                node_obs_shape:int,
                edge_dim:int, graph_aggr:str):
        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads
        self.gnn = GATNet(input_dim=node_obs_shape, edge_dim=edge_dim,
                    num_embeddings=args.num_embeddings,
                    embedding_size=args.embedding_size,
                    hidden_size=args.gnn_hidden_size,
                    num_heads=args.gnn_num_heads,
                    concat_heads=args.gnn_concat_heads,
                    layer_N=args.gnn_layer_N,
                    use_ReLU=args.gnn_use_ReLU,
                    graph_aggr=graph_aggr,
                    global_aggr_type=args.global_aggr_type,
                    embed_hidden_size=args.embed_hidden_size,
                    embed_layer_N=args.embed_layer_N,
                    embed_use_orthogonal=args.use_orthogonal,
                    embed_use_ReLU=args.embed_use_ReLU,
                    embed_use_layerNorm=args.use_feature_normalization,
                    embed_add_self_loop=args.embed_add_self_loop,
                    max_edge_dist=args.max_edge_dist)
        self.out_dim = args.gnn_hidden_size * (args.gnn_num_heads if args.gnn_concat_heads else 1)
        
    def forward(self, node_obs:Tensor, adj:Tensor, agent_id:Tensor):
        batch_size, num_nodes, _ = node_obs.shape
        edge_index, edge_attr = GATNet.process_adj(adj, self.gnn.max_edge_dist)
        # Flatten node_obs
        x = node_obs.view(-1, node_obs.size(-1))
        # Create batch index
        batch = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
        x = self.gnn(data)
        if self.gnn.graph_aggr == 'node':
            x = x.view(batch_size, num_nodes, -1)
            agent_id = agent_id.long()  # Ensure agent_id is long tensor
            x = x.gather(1, agent_id.unsqueeze(-1).expand(-1, -1, x.size(-1))).squeeze(1)
        return x

