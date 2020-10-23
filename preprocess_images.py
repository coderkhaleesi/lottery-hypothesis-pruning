import os
from PIL import Image
import pandas as pd
import albumentations as A
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from __settings__ import img_size


mtcnn = MTCNN(image_size=img_size, post_process=False, select_largest=False, device='cpu').eval()

df = pd.read_excel('SFEW.xlsx')


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


for i in range(len(df)):
    image_src = 'images_raw/' + df.iloc[i, 0][:-4] + '.png'
    image = Image.open(image_src)
    image_cropped = mtcnn(image)
    if image_cropped is None:
        print(image.im.mode)
        image_src = 'images_manual/' + df.iloc[i, 0][:-4] + '.png'
        if not os.path.exists(image_src):
            print(image_src)
            continue
        image = Image.open(image_src)
        image = image.resize((img_size, img_size))
        #image.load()  # required for png.split()

        #background = Image.new("RGB", image.size, (0, 0, 0))
        #background.paste(image, mask=image.split()[3])  # 3 is the alpha channel

        #image = background#background.save('foo.jpg', 'JPEG', quality=80)
        image.save('images/' + df.iloc[i, 0][:-4] + '.png')
    else:
        save_image(normalize(image_cropped), 'images/' + df.iloc[i, 0][:-4] + '.png')

    #transformed = transforms(image=image_cropped)['image']
    #plt.imshow(transformed / transformed.max())
    #plt.show()
