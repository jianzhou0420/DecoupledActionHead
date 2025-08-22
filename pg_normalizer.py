from equi_diffpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
import pickle


with open('equi_diffpo/constant/normalizer_ABCDEFGH.pkl', 'rb') as f:
    normalizer = pickle.load(f)

print(normalizer)
print(type(normalizer))
pass
