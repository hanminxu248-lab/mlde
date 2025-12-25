from transformers import EsmModel

model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model.save_pretrained("./local_esm2")
