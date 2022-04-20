import pytest
from microchannels import Microchannel
from materials import Copper, Air

w_c_data = [50e-6, 100e-6, 150e-6, 200e-6]
w_w_data = [1e-6, 5e-6, 10e-6, 15e-6]
N_data = [499, 242, 159, 118]


# test_data = [
#     # w_c, w_w, N
#     (50e-6, 1e-6, 499),
#     (100e-6, 5e-6, 242),
#     (150e-6, 10e-6, 159),
#     (200e-6, 15e-6, 118),
# ]


@pytest.fixture
def my_mhs():
    copper = Copper()
    air = Air()
    return Microchannel(base=copper, coolant=air)


@pytest.mark.parametrize("w_c,w_w,N", [
    (w_c, w_w, N) for w_c, w_w, N in zip(w_c_data, w_w_data, N_data)
])
def test_n(my_mhs, w_c, w_w, N):
    my_mhs.w_c = w_c
    my_mhs.w_w = w_w
    assert my_mhs.N == N

