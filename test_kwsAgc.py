import numpy as np
from argparse import ArgumentParser, Namespace
from kws_agc import KwsAgc, KwsAgcParams

SAMPLE_RATE: int = 16000


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", dest="input_file", help="Path to input file",
                        required=True, type=str)

    parser.add_argument("-t","--test_file", dest="test_file", type=str, required=True,
                        help="Path to test audio file")

    return parser.parse_args()


args = get_args()

agc = KwsAgc(KwsAgcParams())

with open(args.input_file, 'rb') as h:
    _pcm = np.fromfile(h, dtype=np.int16)
print(_pcm[:10])
print(len(_pcm))
out_pcm = agc(_pcm, rate=16000)

with open(args.test_file, 'rb') as h:
    _pcm_origin = np.fromfile(h, dtype=np.int16)
    print(_pcm_origin[:10])
    print(len(_pcm_origin))
    # FIXME: case problem with float
    assert((out_pcm - _pcm_origin).sum() == 0.0)

print("output data: ")
for el in out_pcm:
    print(el)

print ("PASSED")

