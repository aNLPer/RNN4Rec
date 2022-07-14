import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
# padded sequence padding_value = 0
seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
# seq 中每个序列去掉填充后的实际长度
lens = [2, 1, 3]

# 压缩padded sequence
packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
print(packed.data)
# PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]), sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))

seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)

print(seq_unpacked)
# tensor([[1, 2, 0],
#     	[3, 0, 0],
#       [4, 5, 6]])

print(lens_unpacked)
#tensor([2, 1, 3])


loss = nn.MSELoss(reduction='mean')
input = torch.randn(5)
print(input)
target = torch.randn(5)
print(target)
output = loss(input, target)
print(output)

