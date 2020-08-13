import sys
import argparse 


parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--a', type=str,  default="abc")
parser.add_argument('--b', type=str, dest="b")
parser.add_argument('--c', dest="c", type=int)

print(parser.parse_args(['--a', 'test', '--b', 'val', '--c', '3']))


test='--a test --b val --c 3'

print(parser.parse_args(test.split()))
