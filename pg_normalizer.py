from equi_diffpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
import pickle


with open('/media/jian/ssd4t/DP/first/equi_diffpo/constant/ABCDEFGHIJKL_normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)

print(normalizer)
print(type(normalizer))
pass
