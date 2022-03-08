
# In[ ]:


#@title Generate a Painting
text_query = "an abstract painting with orange triangles" #@param {type:"string"}
num_initial_samples = 200 #@param {type:"slider", min:20, max:1000, step:20}
recombination_amount = 0.333 #@param {type:"slider", min:0, max:0.5, step:0.001}
mutation_amount = 0.25 #@param {type:"slider", min:0, max:0.5, step:0.001}
num_generations = 5 #@param {type:"slider", min:0, max:10, step:1}

text = clip.tokenize(text_query).to(device)
with torch.no_grad():
  query_features = model.encode_text(text)

acc_image_features = torch.empty(0, 512)
acc_sample_images = torch.empty(0, 3, 512, 512)
acc_latents = torch.empty(0, 512)

print ("Generating the initial population, keeping the best 4 out of", num_initial_samples)

num_batches = num_initial_samples // 20

from tqdm.notebook import tqdm
for b in tqdm(range(num_batches)):

  latents = torch.randn(20, 512, device="cuda")
  sample_images, _ = g_ema([latents], truncation=1.0, truncation_latent=None, return_latents=False, randomize_noise=False)
  sample_images_cpu = sample_images.detach().cpu()
  acc_sample_images = torch.vstack((acc_sample_images, sample_images_cpu))
  latents_cpu = latents.detach().cpu()
  acc_latents = torch.vstack((acc_latents, latents_cpu))
  del sample_images
  del latents
  image_input = F.interpolate(sample_images_cpu, input_resolution)

  image_input = np.clip((image_input + 1.0)/2.0, 0, 1)
  image_input -= image_mean[:, None, None]
  image_input /= image_std[:, None, None]

  with torch.no_grad():
    image_features = model.encode_image(image_input).float()
  image_features /= image_features.norm(dim=-1, keepdim=True)
  acc_image_features = torch.vstack((acc_image_features, image_features))

image_similarity = query_features.numpy() @ acc_image_features.numpy().T
image_similarity = image_similarity[0]
image_scores, image_indices = get_top_N_semantic_similarity(image_similarity, N=4)
np_images = acc_sample_images.numpy()
images = []
for i in range(4):
  image = np_images[image_indices[i], :, :, :]
  image = image.transpose(1, 2, 0)
  min = -1 # image.min()
  max = 1 # image.max()
  img = (np.clip(255*(image - min)/(max-min), 0, 255)).astype(int)
  images.append(img)
plt.figure(figsize=(16,24))
columns = 4
for i, image in enumerate(images):
  plt.subplot(len(images) / columns + 1, columns, i + 1)
  plt.axis("off")
  plt.tight_layout()
  plt.imshow(image)
plt.show()

if num_generations > 0:
  print ("Running the GA for", num_generations, "generations")

latents_cpu = acc_latents
generation_indices = image_indices

for g in range(num_generations):
  # start with a blank slate
  new_latents_cpu = torch.empty(0, 512, device="cpu")

  # copy the best four latents (A, B, C, D)
  for i in range(4):
    offspring = latents_cpu[generation_indices[i]]
    new_latents_cpu = torch.vstack((new_latents_cpu, offspring))

  for j in range(4):
    for i in range(4):
      # combine i with j
      offspring = (1-recombination_amount) * new_latents_cpu[i] + recombination_amount * new_latents_cpu[j]

      # and add a little mutation
      offspring += torch.randn(512) * mutation_amount

      # normalize the latent vector
      offspring = F.normalize(offspring, dim=0, p=2) * 100
      new_latents_cpu = torch.vstack((new_latents_cpu, offspring))

  latents_cpu = new_latents_cpu

  acc_sample_images = torch.empty(0, 3, 512, 512)
  new_latents_gpu = latents_cpu.detach().cuda()

  sample_images, _ = g_ema([new_latents_gpu], truncation=1.0, truncation_latent=None, return_latents=False, randomize_noise=False)
  sample_images_cpu = sample_images.detach().cpu()
  acc_sample_images = torch.vstack((acc_sample_images, sample_images_cpu))

  del sample_images
  del new_latents_gpu
  image_input = F.interpolate(sample_images_cpu, input_resolution)

  image_input = np.clip((image_input + 1.0)/2.0, 0, 1)
  image_input -= image_mean[:, None, None]
  image_input /= image_std[:, None, None]

  with torch.no_grad():
    image_features = model.encode_image(image_input).float()
  image_features /= image_features.norm(dim=-1, keepdim=True)

  image_similarity = query_features.numpy() @ image_features.numpy().T
  image_similarity = image_similarity[0]
  image_scores, generation_indices = get_top_N_semantic_similarity(image_similarity, N=4)
  np_images = acc_sample_images.numpy()
  images = []

  # for i in range(4):
  #   image = np_images[generation_indices[i], :, :, :]

  for i in range(20):
    image = np_images[i, :, :, :]
    image = image.transpose(1, 2, 0)
    min = image.min() * 0.9
    max = image.max() * 1.1
    img = (np.clip(255*(image - min)/(max-min), 0, 255)).astype(int)
    images.append(img)
  print("generation", g+1)

  # plt.figure(figsize=(16,24))

  plt.figure(figsize=(8,12))

  columns = 4
  for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(image)
  plt.show()

import numpy as np
import PIL
# convert the image to use floating point
if num_generations > 0:
  img_fp = images[generation_indices[0]].astype(np.float32)
else:
  img_fp = images[0].astype(np.float32)
# stretch the red channel by 0.1% at each end
r_min = np.percentile(img_fp[:,:,0:1], 0.1)
r_max = np.percentile(img_fp[:,:,0:1], 99.9)
img_fp[:,:,0:1] = (img_fp[:,:,0:1]-r_min) * 255.0 / (r_max-r_min)
# stretch the green channel by 0.1% at each end
g_min = np.percentile(img_fp[:,:,1:2], 0.1)
g_max = np.percentile(img_fp[:,:,1:2], 99.9)
img_fp[:,:,1:2] = (img_fp[:,:,1:2]-g_min) * 255.0 / (g_max-g_min)
# stretch the blue channel by 0.1% at each end
b_min = np.percentile(img_fp[:,:,2:3], 0.1)
b_max = np.percentile(img_fp[:,:,2:3], 99.9)
img_fp[:,:,2:3] = (img_fp[:,:,2:3]-b_min) * 255.0 / (b_max-b_min)
# convert the image back to integer, after rounding and clipping
img_int = np.clip(np.round(img_fp), 0, 255).astype(np.uint8)
# convert to the image to PIL and resize to fix the aspect ratio
img_pil=PIL.Image.fromarray(img_int)

print(text_query)
dpi = mpl.rcParams['figure.dpi']
figsize = 512 / float(dpi), 512 / float(dpi)
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])
plt.axis("off")
plt.imshow(img_pil)
plt.show()
