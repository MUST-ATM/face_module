import os, random

class DatumXY:
    """Data instance which defines the basic attributes.

    Args:
        impath_x (str): image path of fake.
        impath_y (str): image path of live.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath_x="", impath_y="", label=-1, domain=-1, classname="", video=""):
        assert isinstance(impath_x, str)
        assert isinstance(impath_y, str)
        self._impath_x = impath_x
        self._impath_y = impath_y
        self._label = label
        self._domain = domain
        self._classname = classname
        self._video = video

    @property
    def impath_x(self):
        return self._impath_x
    @property
    def impath_y(self):
        return self._impath_y
    @property
    def label(self):
        return self._label
    @property
    def domain(self):
        return self._domain
    @property
    def classname(self):
        return self._classname
    @property
    def video(self):
        return self._video

def folder_txt2list(root, protocols, stage, img_index):
    # get data from txt

    with open(os.path.join(root, 'Protocol', protocols, stage + '.txt'), 'r') as f:
        lines = f.readlines()
    lines_ = []
    for line in lines:
        image, label = line.strip().split(' ')
        relative_image_path = image.lstrip('/')
        
        if img_index == "random":
            # Get directory path
            dir_path = os.path.join(root, relative_image_path)
            # Check if the path is a directory and list all images
            if os.path.isdir(dir_path):
                available_images = os.listdir(dir_path)
                if available_images:
                    # Randomly select an image
                    selected_image = random.choice(available_images)
                    impath = os.path.join(dir_path, selected_image)
                else:
                    continue  # Skip if no images are found
            else:
                continue  # Skip if it's not a directory
        else:
            # Use the provided img_index
            impath = os.path.join(root, relative_image_path, img_index)
            
        # Create the tuple to append to the list
        if stage == 'train':
            lines_.append((impath, int(label)))
        else:
            pairs = []
            pairs.append(impath)
            pairs.append(int(label))
            lines_.append(tuple(pairs))

    # data balance to 1:1
    if stage == 'train':
        lives, fakes = [], []
        for line in lines_:
            impath, label = line
            if label == 0:
                lives.append(line)
            else:
                fakes.append(line)
        insert = len(fakes) - len(lives)
        if insert < 0:
            insert = -insert
            for _ in range(insert):
                fakes.append(random.choice(fakes))
        elif insert > 0:
            for _ in range(insert):
                lives.append(random.choice(lives))
        else:
            pass
        assert len(lives) == len(fakes)
        return lives, fakes
    else:
        return lines_
    
def img_txt2list(root, protocols, stage):
    # get data from txt
    with open(os.path.join(root, 'Protocol', protocols, stage + '.txt')) as f:
        lines = f.readlines()
        f.close()
    lines_ = []
    for line in lines:
        image, label = line.strip().split(' ')
        if stage == 'train':
            impath = os.path.join(root, image.lstrip('/'))
            lines_.append((impath, int(label)))
        else:
            pairs = []
            pairs.append(os.path.join(root, image.lstrip('/')))
            pairs.append(int(label))
            lines_.append(tuple(pairs))

    # data balance to 1:1
    if stage == 'train':
        lives, fakes = [], []
        for line in lines_:
            impath, label = line
            if label == 0:
                lives.append(line)
            else:
                fakes.append(line)
        insert = len(fakes) - len(lives)
        if insert < 0:
            insert = -insert
            for _ in range(insert):
                fakes.append(random.choice(fakes))
        elif insert > 0:
            for _ in range(insert):
                lives.append(random.choice(lives))
        else:
            pass
        assert len(lives) == len(fakes)
        return lives, fakes
    else:
        return lines_

def read_data(data_root, input_domain, protocols, split, txt_type, img_index='00.png'):
    items = []
    if split == 'train':
        if txt_type == 'img':
            lives_list, fakes_list = img_txt2list(data_root, protocols, split)
        elif txt_type == 'folder':
            lives_list, fakes_list = folder_txt2list(data_root, protocols, split, img_index)
        for i in range(len(fakes_list)):
            item = DatumXY(
                impath_x=fakes_list[i][0],
                impath_y=lives_list[i][0],
                domain=input_domain
            )
            items.append(item)
        print('Load {} {}={} pairs'.format(input_domain, split, len(lives_list)))
        return items
    else:
        if txt_type == 'img':
            val_data_list = img_txt2list(data_root, protocols, split)
        elif txt_type == 'folder':
            val_data_list = folder_txt2list(data_root, protocols, split, img_index)
        for impath, label in val_data_list:
            item = DatumXY(
                impath_x=impath,
                label=label
            )
            items.append(item)
        print('Load {} {}={} images'.format(input_domain, split, len(val_data_list)))
        return items

