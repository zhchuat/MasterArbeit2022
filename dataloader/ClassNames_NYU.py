import numpy as np

CLASS_NAMES_14 = ['void',
                      'bed', 'books', 'ceiling', 'chair', 'floor', 'furniture',
                      'objects', 'picture', 'sofa', 'table', 'tv', 'wall',
                      'window','LIMO']
CLASS_NAMES_40 = ['void',
                      'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                      'table', 'door', 'window', 'bookshelf', 'picture',
                      'counter', 'blinds', 'desk', 'shelves', 'curtain',
                      'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
                      'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag',
                      'otherstructure', 'otherfurniture', 'otherprop']

ColorsMap = [
            (0, 0, 0), 
            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), 
            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), 
            (192, 128, 0), (64, 0, 128), (192, 0, 128),(120,100,50)
            ]

# def get_colormap(n):
#     def bitget(byteval, idx):
#         return (byteval & (1 << idx)) != 0

#     colormap = np.zeros((n, 3), dtype='uint8')
#     for i in range(n):
#         r = g = b = 0
#         c = i
#         for j in range(8):
#             r = r | (bitget(c, 0) << 7-j)
#             g = g | (bitget(c, 1) << 7-j)
#             b = b | (bitget(c, 2) << 7-j)
#             c = c >> 3

#         colormap[i] = np.array([r, g, b])

#     return colormap

# CLASS_NAMES_13 = get_colormap(1+13).tolist()
# print(CLASS_NAMES_13)


