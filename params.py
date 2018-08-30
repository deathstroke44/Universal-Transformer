import argparse

model_dim = 64
batch_size = 64
dropout = 0
h = 8
t_steps = 4

"""
Overriding Hyper-parameter using Argument Parser
"""
# Getting Default Hyper Parameter
__params = [key for key, value in locals().items() if key[:2] != "__" and key != "argparse"]

# Parsing Hyper Parameter
__parser = argparse.ArgumentParser(description='Clova Duplex Hyper Params')
for param in __params:
    __parser.add_argument('--' + param, type=type(locals()[param]), default=eval(param), dest=param)
__args = __parser.parse_args()

# Overriding Default Hyper Parameter to Argument Parameter
for param in __params: locals()[param] = __args.__getattribute__(param)
config = vars(__args)

print("Hyper-param", config)