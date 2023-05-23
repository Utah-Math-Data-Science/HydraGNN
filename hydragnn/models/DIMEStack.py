"""
DimeNet
========
Directional message passing neural network
for molecular graphs. The convolutional
layer uses spherical and radial basis
functions to perform message passing.

In particular this message passing layer
relies on the angle formed by the triplet
of incomming and outgoing messages.

The three key components of this network are
outlined below. In particular, the convolutional
network that is used for the message passing
the triplet function that generates to/from
information for angular values, and finally
the radial basis embedding that is used to
include radial basis information.

"""
from math import sqrt

from typing import Callable, Optional, Tuple
from torch_geometric.typing import SparseTensor

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, SiLU

from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    SphericalBasisLayer,
    ResidualLayer,
)
from torch_geometric.utils import scatter

from .Base import Base


class DIMEStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        num_bilinear,
        num_radial,
        num_spherical,
        radius,
        envelope_exponent,
        num_before_skip,
        num_after_skip,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.radius = radius

        self.rbf = BesselBasisLayer(num_radial, radius, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, radius, envelope_exponent
        )

        self.interact = Interaction(
            hidden_channels=self.hidden_dim,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
        )

    def _conv_args(self, data):
        conv_args = {"edge_index": data.edge_index}
        assert (
            data.pos is not None
        ), "DimeNet requires node positions (data.pos) to be set."
        conv_args.update({"pos": data.pos})
        return conv_args

    def forward(self, z, pos, edge_index):
        z = z.to(torch.long)
        # edge_index = radius_graph(pos, r=self.radius, batch=batch,
        #                           max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0)
        )
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        z = self.interact(z, rbf, sbf, idx_kj, idx_ji)
        # z = z + output_block(x, rbf, i, num_nodes=pos.size(0))

        return z


class Interaction(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
    ):
        super().__init__()
        self.act = SiLU()

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear, bias=False)

        # Dense transformations of input messages.
        self.lin_from = Linear(hidden_channels, hidden_channels)
        self.lin_to = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.Tensor(hidden_channels, num_bilinear, hidden_channels)
        )

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, SiLU()) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, SiLU()) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_from.weight, scale=2.0)
        self.lin_from.bias.data.fill_(0)
        glorot_orthogonal(self.lin_to.weight, scale=2.0)
        self.lin_to.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        radial_basis: Tensor,
        spherical_basis: Tensor,
        edge_index_from: Tensor,
        edge_index_to: Tensor,
    ) -> Tensor:
        radial_basis = self.lin_rbf(radial_basis)
        spherical_basis = self.lin_sbf(spherical_basis)

        x_kj = self.act(self.lin_from(x))
        x_kj = x_kj * radial_basis
        x_kj = torch.einsum(
            "wj,wl,ijl->wi", spherical_basis, x_kj[edge_index_from], self.W
        )
        x_kj = scatter(
            x_kj, edge_index_to, dim=0, dim_size=x.size(0), reduce="sum"
        )  # message passing

        x_ji = self.act(self.lin_to(x))
        h = (
            x_ji + x_kj
        )  # aggregates my learned message and my from messages to the next neighbor

        for (
            layer
        ) in (
            self.layers_before_skip
        ):  # this added resnet is not actually doing any message passing and is an interesting addition
            h = layer(h)
        h = (
            self.act(self.lin(h)) + x
        )  # incorporates a residual connection to the input feature
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


"""
Triplets
---------
Generates to/from edge_indices for
angle generating purposes.

"""


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


"""
EmbeddingBlock
---------------
An embedding block that utilizes the
radial basis function and the to/from
information in the embedding by
concatentating the to/from nodes with
the radial basis functions.

"""


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
