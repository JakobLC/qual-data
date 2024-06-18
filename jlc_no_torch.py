import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from matplotlib.patheffects import withStroke
from skimage.measure import find_contours
from PIL import Image
import io
import scipy.ndimage as nd
from pprint import pprint

large_pallete = [  0,   0,   0,  23, 190, 207, 255, 127,  14, 214,  39,  40, 152,
       251, 152,   0,   0, 142, 148, 103, 189, 220, 220,   0, 140,  86,
        75, 107, 142,  35, 220,  20,  60, 255,   0,   0, 255, 255,  90,
       102, 102, 156,  31, 119, 180,   0,   0,  70, 119,  11,  32, 205,
       255,  50,   0,  80, 100, 250, 170,  30,   0,   0, 230, 244,  35,
       232, 227, 119, 194, 255, 220,  80,  44, 160,  44, 190, 153, 153,
       128,  64, 128,   0,  60, 100]

largest_pallete = [  0,   0,   0]+sum([large_pallete[3:] for _ in range(255*3//len(large_pallete)+2)],[])[:255*3]

large_colors = np.array(large_pallete[3:]).reshape(-1, 3)
largest_colors = np.array(largest_pallete[3:]).reshape(-1, 3)

def darker_color(x,power=2,mult=0.5):
    assert isinstance(x,np.ndarray), "darker_color expects an np.ndarray"
    is_int_type = x.dtype in [np.uint8,np.uint16,np.int8,np.int16,np.int32,np.int64]
    if is_int_type:
        return np.round(255*darker_color(x/255,power=power,mult=mult)).astype(np.uint8)
    else:
        return np.clip(x**power*mult,0,1)

def get_mask(mask_vol,idx,onehot=False,onehot_dim=-1):
    if onehot:
        slice_idx = [slice(None) for _ in range(len(mask_vol.shape))]
        slice_idx[onehot_dim] = idx
        return np.expand_dims(mask_vol[tuple(slice_idx)],onehot_dim)
    else:
        return (mask_vol==idx).astype(float)

def mask_overlay_smooth(image,
                        mask,
                        pallete=None,
                        pixel_mult=1,
                        class_names=None,
                        show_border=False,
                        border_color="darker",
                        border_alpha=1.0,
                        mask_alpha=0.4,
                        dont_show_idx=[255],
                        fontsize=12,
                        fontsize_fixed_image_size=None,
                        text_color="class",
                        text_alpha=1.0,
                        text_outline_instead_of_highlight=True,
                        set_lims=True,
                        class_names_distance_transform=True):
    """
    Author: Jakob Lønborg Christensen (jloch@dtu.dk), 2024
    A function to visualize a segmentation on top of an image. The function
    can handle both one-hot encoded masks and integer masks.

    Parameters
    ----------
    image : np.ndarray
        Image to overlay the mask on. If the image is uint8 it is converted to
        float and divided by 255. The returned image is uint8 if the input is.
    mask : np.ndarray
        Mask to overlay on the image. If it has a third non-trivial dimension
        it is assumed to be one-hot encoded. If the mask is onehot encoded
        it can be float format. If the mask is shaped (H,W) or (H,W,1) it 
        should be integer format.
    pallete : np.ndarray, optional
        A pallete of colors to use for the different classes. The default is
        None, which means the default pallete is used. The default pallete is
        a set of nice colors with class 0 being black as the only one.
    pixel_mult : int, optional
        Multiplier for the number of pixels in each spatial dimension wrt.
        the original image. The default is 1, meaning the image.shape=(H,W).
        Otherwise the image.shape=(H*pixel_mult,W*pixel_mult). This is useful
        for making things drawn on top of the image (e.g. text, borders) more
        visible - especially for small images.
    class_names : dict, optional
        A dictionary mapping class indices to class names. The default is None.
        Text is drawn on top of the image with the class names if passed.
    show_border : bool, optional
        Should the border of the mask be shown. The default is False.
    border_color : str, optional
        Color of the border. The default is "darker". The same color is used
        for all classes if the border_color!="darker".
    border_alpha : float, optional
        The opacity of the border. The default is 1.0.
    mask_alpha : float, optional
        Opacity of the mask. The default is 0.4.
    dont_show_idx : list, optional
        List of class indices not to show masks from. The default is [255], usually
        representing padding for uint8 segmentations.
    fontsize : int, optional
        Font size of the class names. The default is 12.
    fontsize_fixed_image_size : int or float, optional
        If passed, the fontsize is rescaled with the ratio between the passed
        image's size and this variable. E.g. if the fontsize looks good for
        224x224 images, then pass fontsize_fixed_image_size=224. The default is
        None (no rescaling when)
    text_color : str, optional
        Color of the text. The default is "class", meaning the color of the
        class of the used pallete. If text_color is not "class" it should be
        a matplotlib color-like variable.
    text_alpha : float, optional
        Opacity of the text. The default is 1.0. Only relevant if class_names
        is not None.
    text_outline_instead_of_highlight : bool, optional
        If True the text will have an outline, otherwise a square 
        highlight. The default is True. Only relevant if class_names is not
        None.
    set_lims : bool, optional
        Should the limits of the image be set. The default is True. This
        option is only usually relevant if e.g. text is added to the image
        and it renders outside the limits of the image.
    class_names_distance_transform : bool, optional
        Should the class names be placed at the point furthest away from
        other classes (True), otherwise the center of mass is used (False).
        The default is True.
    Returns
    -------
    image_colored : np.ndarray
        The image with the mask overlayed, aswell as other rendering options
        which were set.

    """
    assert isinstance(image,np.ndarray)
    assert isinstance(mask,np.ndarray)
    assert 2<=len(image.shape)<=3, "image must have 2 or 3 dimensions"
    assert 2<=len(mask.shape)<=3, "mask must have 2 or 3 dimensions"
    assert image.shape[:2]==mask.shape[:2], "image and mask must have the same shape. Expecting shapes in (H,W,C) format, or (H,W) format"
    if pallete is None:
        pallete = np.concatenate([np.array([[0,0,0]]),largest_colors],axis=0)
    if image.dtype==np.uint8:
        was_uint8 = True
        image = image.astype(float)/255
    else:
        was_uint8 = False
    if len(mask.shape)==2:
        onehot = False
        n = mask.max()+1
        uq = np.unique(mask).tolist()
        mask = np.expand_dims(mask.copy(),-1)
        assert mask.dtype in [np.uint8,np.int32,np.int64], "mask must be integer format if it is not one-hot encoded"
    else:
        if mask.shape[2]==1:
            onehot = False
            n = mask.max()+1
            uq = np.unique(mask).tolist()
            assert mask.dtype in [np.uint8,np.int32,np.int64], "mask must be integer format if it is not one-hot encoded"
        else:
            assert mask.shape[2]>1, "If mask is (H,W,C) then C>=1 is required"
            onehot = True
            n = mask.shape[2]
            uq = np.arange(n).tolist()
    image_colored = image.copy()
    if len(image_colored.shape)==2:
        image_colored = np.expand_dims(image_colored,-1)
    #make rgb
    if image_colored.shape[-1]==1:
        image_colored = np.repeat(image_colored,3,axis=-1)
    show_idx = [i for i in uq if (not i in dont_show_idx)]
    for i in show_idx:
        reshaped_color = pallete[i].reshape([1,1,3])/255
        mask_coef = mask_alpha*get_mask(mask,i,onehot=onehot)
        image_coef = 1-mask_coef
        image_colored = image_colored*image_coef+reshaped_color*mask_coef
    if class_names is not None:
        assert isinstance(class_names,dict), "class_names must be a dictionary that maps class indices to class names"
        for i in uq:
            assert i in class_names.keys(), f"class_names must have a key for each class index, found i={i} not in class_names.keys()"
    assert isinstance(pixel_mult,int), "pixel_mult must be an integer"
    
    if pixel_mult>1:
        image_colored = cv2.resize(image_colored,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_NEAREST)
    
    image_colored = np.clip(image_colored,0,1)
    if show_border or (class_names is not None):
        image_colored = (image_colored*255).astype(np.uint8)
        h,w = image_colored.shape[:2]
        with RenderMatplotlibAxis(h,w,set_lims=set_lims) as ax:
            plt.imshow(image_colored)
            for i in show_idx:
                mask_coef = get_mask(mask,i,onehot=onehot)
                if pixel_mult>1:
                    mask_coef = cv2.resize(mask_coef,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_LANCZOS4)
                else:
                    mask_coef = mask_coef.reshape(h,w)
                if show_border:                    
                    curves = find_contours(mask_coef, 0.5)
                    if border_color=="darker":
                        border_color_i = darker_color(pallete[i]/255)
                    else:
                        border_color_i = border_color
                    k = 0
                    for curve in curves:
                        plt.plot(curve[:, 1], curve[:, 0], linewidth=1, color=border_color_i, alpha=border_alpha)
                        k += 1

                if class_names is not None:
                    t = class_names[i]
                    if len(t)>0:
                        if class_names_distance_transform:
                            dist = distance_transform_edt_border(mask_coef)
                            y,x = np.unravel_index(np.argmax(dist),dist.shape)
                        else:
                            #center of mass
                            x = np.mean(np.where(mask_coef>0)[1])
                            y = np.mean(np.where(mask_coef>0)[0])
                        if text_color=="class":
                            text_color_i = pallete[i]/255
                        else:
                            text_color_i = text_color
                        avg_of_HW = (h+w)/2
                        if fontsize_fixed_image_size is None:
                            font_rescaler = 1
                        else:
                            assert isinstance(fontsize_fixed_image_size,(int,float)), "fontsize_fixed_image_size must be an integer or float"
                            font_rescaler = avg_of_HW/fontsize_fixed_image_size
                        text_kwargs = {"fontsize": int(round(fontsize*pixel_mult*font_rescaler)),
                                       "color": text_color_i,
                                       "alpha": text_alpha}
                        col_bg = "black" if np.mean(text_color_i)>0.5 else "white"             
                        t = plt.text(x,y,t,**text_kwargs)
                        if text_outline_instead_of_highlight:
                            t.set_path_effects([withStroke(linewidth=3, foreground=col_bg)])
                        else:
                            t.set_bbox(dict(facecolor=col_bg, alpha=text_alpha, linewidth=0))
        image_colored = ax.image
    else:
        if was_uint8: 
            image_colored = (image_colored*255).astype(np.uint8)
    return image_colored

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]

class RenderMatplotlibAxis:
    """
    Author: Jakob Lønborg Christensen (jloch@dtu.dk), 2024
    Class for rendering things on top of an image using matplotlib.
    E.g. if you want to write text on top of an image and then save
    the actual image with the text on top of it.
    The class is used like the following example
    image = np.random.rand(100,100)
    with RenderMatplotlibAxis(100,100) as ax:
        plt.imshow(image)
        plt.text(10,50,"Hello World")
    image_with_text = ax.image
    """
    def __init__(self, height, width=None, with_axis=False, set_lims=False, with_alpha=False, dpi=100):
        if (width is None) and isinstance(height, (tuple, list)):
            #height is a shape
            height,width = height[:2]
        elif (width is None) and isinstance(height, np.ndarray):
            #height is an image
            height,width = height.shape[:2]
        elif width is None:
            width = height
        self.with_alpha = with_alpha
        self.width = width
        self.height = height
        self.dpi = dpi
        self.old_backend = matplotlib.rcParams['backend']
        self.old_dpi = matplotlib.rcParams['figure.dpi']
        self.fig = None
        self.ax = None
        self._image = None
        self.with_axis = with_axis
        self.set_lims = set_lims

    @property
    def image(self):
        return self._image[:,:,:(3+int(self.with_alpha))]

    def __enter__(self):
        matplotlib.rcParams['figure.dpi'] = self.dpi
        matplotlib.use('Agg')
        figsize = (self.width/self.dpi, self.height/self.dpi)
        self.fig = plt.figure(figsize=figsize,dpi=self.dpi)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        if not self.with_axis:
            self.ax.set_frame_on(False)
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
        self.fig.add_axes(self.ax)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If no exception occurred, save the image to the _image property
            if self.set_lims:
                self.ax.set_xlim(-0.5, self.width-0.5)
                self.ax.set_ylim(self.height-0.5, -0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0, dpi=self.dpi)
            buf.seek(0)
            self._image = np.array(Image.open(buf))

        plt.close(self.fig)
        matplotlib.use(self.old_backend)
        matplotlib.rcParams['figure.dpi'] = self.old_dpi

def is_type_for_dot_shape(item):
    return isinstance(item,np.ndarray)

def is_deepest_expand(x,expand_deepest,max_expand):
    if expand_deepest:
        if isinstance(x,str):
            if len(x)<=max_expand:
                out = True
            else:
                out = False
        elif isinstance(x,(int,float)):
            out = True
        else:
            out = False
    else:
        out = False
    return out

def is_type_for_recursion(item,m=20):
    out = False
    if isinstance(item,(list,dict,tuple)):
        if len(item)<=m:
            out = True
    return out

def reduce_baseline(x):
    if hasattr(x,"__len__"):
        lenx = len(x)
    else:
        lenx = -1
    return f"<{type(x).__name__}>len{lenx}"

def fancy_shape(item):
    assert is_type_for_dot_shape(item)
    out = f"np.Size({list(item.shape)})"
    return out

def shaprint(x, max_recursions=5, max_expand=20, first_only=False,do_pprint=True,do_print=False,return_str=False, expand_deepest=False):
    """
    Author: Jakob Lønborg Christensen (jloch@dtu.dk), 2024
    Prints almost any object as a nested structure of shapes and lengths.
    Example:
    strange_object = {"a":np.random.rand(3,4,5),"b": [np.random.rand(3,4,5) for _ in range(3)],"c": {"d": [torch.rand(3,4,5),[[1,2,3],[4,5,6]]]}}
    shaprint(strange_object)
    """
    kwargs = {"max_recursions":max_recursions,
              "max_expand":max_expand,
              "first_only":first_only,
              "do_pprint": False,
              "do_print": False,
              "return_str": True,
              "expand_deepest":expand_deepest}
    m = float("inf") if first_only else max_expand
    if is_type_for_dot_shape(x):
        out = fancy_shape(x)
    elif is_deepest_expand(x,expand_deepest,max_expand):
        out = x
    elif is_type_for_recursion(x,m):
        if kwargs["max_recursions"]<=0:
            out = reduce_baseline(x)
        else:
            kwargs["max_recursions"] -= 1
            if isinstance(x,list):
                if first_only:
                    out = [shaprint(x[0],**kwargs)]
                else:
                    out = [shaprint(a,**kwargs) for a in x]
            elif isinstance(x,dict):
                if first_only:
                    k0 = list(x.keys())[0]
                    out = {k0: shaprint(x[k0],**kwargs)}
                else:
                    out = {k:shaprint(v,**kwargs) for k,v in x.items()}
            elif isinstance(x,tuple):
                if first_only:
                    out = tuple([shaprint(x[0],**kwargs)])
                else:
                    out = tuple([shaprint(a,**kwargs) for a in x])
    else:    
        out = reduce_baseline(x)
    if do_pprint:
        pprint(out)
    if do_print:
        print(out)
    if return_str:
        return out
        

def montage(arr,
            maintain_aspect=True,
            reshape=True,
            text=None,
            return_im=False,
            imshow=True,
            reshape_size=None,
            n_col=None,
            n_row=None,
            padding=0,
            padding_color=0,
            rows_first=True,
            figsize_per_pixel=1/100,
            text_color=[1,0,0],
            text_size=10,
            create_figure=True):
    """
    Author: Jakob Lønborg Christensen (jloch@dtu.dk), 2024
    Displays and returns an montage of images from a list or 
    list of lists of images.

    Parameters
    ----------
    arr : list
        A list or list of lists containing images (np.arrays) of shape 
        (d1,d2), (d1,d2,1), (d1,d2,3) or (d1,d2,4). If arr is a list of lists
        then the first list dimensions is vertical and second is horizontal.
        If there is only one list dimensions then the list will be put in an
        appropriate 2D grid of images. The input can also be a 5D or 4D 
        np.array and in this case the first two dimensions are intepreted 
        the same way as if they were a list. Even if the 5th channel dimension
        is size 1 it has to in included in this case.
    maintain_aspect : boolean, optional
        Should image aspect ratios be maintained. Only relevant if 
        reshape=True. The default is True.
    reshape : boolean, optional
        Should images be reshaped to better fit the montage image. The default 
        is True.
    imshow : boolean, optional
        Should plt.imshow() be used inside the function. The default is True.
    reshape_size : array-like, optional
        2 element list or array like variable. Specifies the number of pixels 
        in the first dim (vertical) and second dim (horizontal) per image in
        the resulting concatenated image
        The default is None.
    n_col : int, optional
        Number of columns the montage will contain.
        The default is None.
    n_row : int, optional
        Number of rows the montage will contain.
        The default is None.
    padding : int or [int,int], optional
        Number of added rows/columns of padding to each image. If an int is
        given the same horizontal and vertical padding is used. If a list is
        given then the first index is the number of vertical padding pixels and
        the second index is the number of horizontal padding pixels. 
        The default is None.
    padding_color : float or int
        The color of the used padding. The default is black (0).
    rows_first : bool
        If True and a single list is given as arr then the images will first
        be filled into row 0 and then row 1, etc. Otherwise columns will be
        filled first. The default is True.
    figsize_per_pixel : float
        How large a figure to render if imshow=True, in relation to pixels.
        Defaults to 1/100.
    text_color : matplotlib color-like
        color of text to write on top of images. Defaults to red ([1,0,0]).
    text_size : float or int
        Size of text to write on top of images. Defaults to 10.
    create_figure : bool
        Should plt.figure() be called when imshow is True? Defaults to True.
    Returns
    -------
    im_cat : np.array
        Concatenated montage image.
        
    
    Example
    -------
    montage(np.random.rand(2,3,4,5,3),reshape_size=(40,50))

    """
    # commented out to avoid torch dependency
    #if torch.is_tensor(arr):
    #    assert len(arr.shape)==4, "torch tensor must have at 4 dims, formatted as (n_images,channels,H,W)"
    #    arr = arr.detach().cpu().clone().permute(0,2,3,1).numpy()
    if isinstance(arr,np.ndarray):
        if len(arr.shape)==4:
            arr = [arr[i] for i in range(arr.shape[0])]
        elif len(arr.shape)==5:
            n1 = arr.shape[0]
            n2 = arr.shape[1]
            arr = [[arr[i,j] for j in range(arr.shape[1])]
                   for i in range(arr.shape[0])]
        else:
            raise ValueError("Cannot input np.ndarray with less than 4 dims")
    
    if isinstance(arr[0],np.ndarray): #if arr is a list or 4d np.ndarray
        if (n_col is None) and (n_row is None):
            n1 = np.floor(len(arr)**0.5).astype(int)
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is None) and (n_row is not None):
            n1 = n_row
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is not None) and (n_row is None):
            n2 = n_col
            n1 = np.ceil(len(arr)/n2).astype(int)
        elif (n_col is not None) and (n_row is not None):
            assert n_col*n_row>=len(arr), "number of columns/rows too small for number of images"
            n1 = n_row
            n2 = n_col
        
        if rows_first:
            arr2 = []
            for i in range(n1):
                arr2.append([])
                for j in range(n2):
                    ii = n2*i+j
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        else:
            arr2 = [[] for _ in range(n1)]
            for j in range(n2):
                for i in range(n1):
                    ii = i+j*n1
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        arr = arr2
    if n_row is None:
        n1 = len(arr)
    else:
        n1 = n_row
        
    n2_list = [len(arr[i]) for i in range(n1)]
    if n_col is None:
        n2 = max(n2_list)
    else:
        n2 = n_col
        
    idx = []
    for i in range(n1):
        idx.extend([[i,j] for j in range(n2_list[i])])
    n = len(idx)
    idx = np.array(idx)
    
    N = list(range(n))
    I = idx[:,0].tolist()
    J = idx[:,1].tolist()
    
    D1 = np.zeros(n,dtype=int)
    D2 = np.zeros(n,dtype=int)
    aspect = np.zeros(n)
    im = np.zeros((32,32,3))
    channels = 1
    for n,i,j in zip(N,I,J): 
        if arr[i][j] is None:#image is replaced with zeros of the same size as the previous image
            arr[i][j] = np.zeros_like(im)
        else:
            assert isinstance(arr[i][j],np.ndarray), "images in arr must be np.ndarrays (or None for a zero-image)"
        im = arr[i][j]
        
        D1[n] = im.shape[0]
        D2[n] = im.shape[1]
        if len(im.shape)>2:
            channels = max(channels,im.shape[2])
            assert im.shape[2] in [1,3,4]
            assert len(im.shape)<=3
    aspect = D1/D2
    if reshape_size is not None:
        G1 = reshape_size[0]
        G2 = reshape_size[1]
    else:
        if reshape:
            G2 = int(np.ceil(D2.mean()))
            G1 = int(np.round(G2*aspect.mean()))
        else:
            G1 = int(D1.max())
            G2 = int(D2.max())
    if padding is not None:
        if isinstance(padding,int):
            padding = [padding,padding]
    else:
        padding = [0,0]
        
    p1 = padding[0]
    p2 = padding[1]
    G11 = G1+p1*2
    G22 = G2+p2*2
    
    
    im_cat_size = [G11*n1,G22*n2]

    im_cat_size.append(channels)
    im_cat = np.zeros(im_cat_size)
    if channels==4:
        im_cat[:,:,3] = 1

    for n,i,j in zip(N,I,J): 
        im = arr[i][j]
        if issubclass(im.dtype.type, np.integer):
            im = im.astype(float)/255
        if not reshape:
            d1 = D1[n]
            d2 = D2[n]
        else:
            z_d1 = G1/D1[n]
            z_d2 = G2/D2[n]
            if maintain_aspect:
                z = [min(z_d1,z_d2),min(z_d1,z_d2),1][:len(im.shape)]
            else:
                z = [z_d1,z_d2,1][:len(im.shape)]
            im = nd.zoom(im,z)
            d1 = im.shape[0]
            d2 = im.shape[1]
            
        if len(im.shape)==3:
            im = np.pad(im,((p1,p1),(p2,p2),(0,0)),constant_values=padding_color)
        elif len(im.shape)==2:
            im = np.pad(im,((p1,p1),(p2,p2)),constant_values=padding_color)
        else:
            raise ValueError("images in arr must have 2 or 3 dims")
            
        d = (G1-d1)/2
        idx_d1 = slice(int(np.floor(d))+i*G11,G11-int(np.ceil(d))+i*G11)
        d = (G2-d2)/2
        idx_d2 = slice(int(np.floor(d))+j*G22,G22-int(np.ceil(d))+j*G22)
        
        if len(im.shape)>2:
            im_c = im.shape[2]
        else:
            im_c = 1
            im = im[:,:,None]
            
        if im_c<channels:
            if channels>=3 and im_c==1:
                if len(im.shape)>2:
                    im = im[:,:,0]
                im = np.stack([im]*3,axis=2)
            if channels==4 and im_c<4:
                im = np.concatenate([im]+[np.ones((im.shape[0],im.shape[1],1))],axis=2)
        im_cat[idx_d1,idx_d2,:] = im
    #im_cat = np.clip(im_cat,0,1)
    if imshow:
        if create_figure:
            plt.figure(figsize=(figsize_per_pixel*im_cat.shape[1],figsize_per_pixel*im_cat.shape[0]))
        
        is_rgb = channels>=3
        if is_rgb:
            plt.imshow(np.clip(im_cat,0,1),vmin=0,vmax=1)
        else:
            plt.imshow(im_cat,cmap="gray")

        if text is not None:
            #max_text_len = max([max(list(map(len,str(t).split("\n")))) for t in text])
            #text_size = 10#*G22/max_text_len*figsize_per_pixel #42.85714=6*16/224/0.01
            for i,j,t in zip(I,J,text):
                dt1 = p1+G11*i
                dt2 = p2+G22*j
                plt.text(x=dt2,y=dt1,s=str(t),color=text_color,va="top",ha="left",size=text_size)

    if return_im:
        return im_cat
    

class zoom:
    """
    Author: Jakob Lønborg Christensen (jloch@dtu.dk), 2024
    A class for zooming in and out of a matplotlib plot when scrolling,
    as well as dragging the plot around.
    The class is meant to be used with %matplotlib qt5 in a jupyter 
    notebook (not inline). It is used, simply by creating an instance 
    of the class after showing an image. Example usage:
    plt.imshow(np.random.rand(100,100))
    jlc.zoom()
    """
    def __init__(self, ax=None):
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.cid_scroll = ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.zoom_factor = 1.2
        self.dragging = False
        self.prev_x = None
        self.prev_y = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        xdata, ydata = event.xdata, event.ydata

        if event.button == 'up':
            self.zoom_in(xdata, ydata)
        elif event.button == 'down':
            self.zoom_out(xdata, ydata)

        self.ax.figure.canvas.draw()
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            self.dragging = True
            self.prev_x = event.xdata
            self.prev_y = event.ydata

    def on_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.prev_x = None
            self.prev_y = None

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.dragging:
            if self.prev_x is not None and self.prev_y is not None:
                dx = event.xdata - self.prev_x
                dy = event.ydata - self.prev_y
                self.translate(dx, dy)
                self.ax.figure.canvas.draw()
            self.prev_x, self.prev_y = self.ax.transData.inverted().transform((event.x, event.y))

    def zoom_in(self, x, y):
        self.update(x, y, 1 / self.zoom_factor)

    def zoom_out(self, x, y):
        self.update(x, y, self.zoom_factor)

    def translate(self, dx, dy):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xlim = xlim - dx
        ylim = ylim - dy

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.update()

    def update(self, x=None, y=None, factor=1.0):
        if x is not None and y is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            new_width = (xlim[1] - xlim[0]) * factor
            new_height = (ylim[1] - ylim[0]) * factor

            x_ratio = (x - xlim[0]) / (xlim[1] - xlim[0])
            y_ratio = (y - ylim[0]) / (ylim[1] - ylim[0])

            xlim = x - new_width * x_ratio, x + new_width * (1 - x_ratio)
            ylim = y - new_height * y_ratio, y + new_height * (1 - y_ratio)

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.ax.figure.canvas.draw()

