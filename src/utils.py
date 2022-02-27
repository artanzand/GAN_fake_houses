from selenium import webdriver
import os
import urllib
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from PIL import Image, ImageChops


def google_downloader(search, num_images, PATH='chromedriver'):
    """
    Download images from google using selenium and store them in a local folder

    Parameters
    ----------
    search : str
        decription of the image to search in google

    num_images : int
        The number of images to download

    PATH : str
        chromedriver.exe path in the local machine

    Returns
    -------
    Image.jpg : 
        returns dowloaded images from google
    Examples
    --------
    >>> google_downloader('houses vancouver', 4)
    """
    
    url_pre = "https://www.google.com/search?q="
    url_post = "&tbm=isch&ved=2ahUKEwiSpqes5_P1AhXwADQIHdQaAn0Q2-cCegQIABAA&oq=house&gs_lcp=CgNpbWcQA1DIFVi6cWCZcmgAcAB4AIABVYgBxgOSAQE2mAEAoAEBqgELZ3dzLXdpei1pbWewAQDAAQE&sclient=img&ei=nFIEYpLoE_CB0PEP1LWI6Ac&bih=802&biw=1707"
    folder = "train_images"
    
    # create folder if does not exists
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    search_url = url_pre + search + url_post
    driver = webdriver.Chrome(executable_path=PATH)
    driver.get(search_url)
    
    # scroll the website
    scroll = 0 
    for i in range(500):
        driver.execute_script("window.scrollBy("+ str(scroll)+ ",+2000);")
        scroll += 2000
    
    # find elements class "rg_i"
    elements = driver.find_elements(By.CLASS_NAME,'rg_i')
    
    # download images
    for i,e in enumerate(elements):
        if i <= num_images:
            src = e.get_attribute('src')
            try :
                if src != None:
                    urllib.request.urlretrieve(src, filename=os.path.join(folder,'image_'+str(i)+'.jpg'))
                    print("Downloaded image", search, i+1)
            except TypeError:
                print('Not Downloaded!') 
    
    driver.quit()
    
    # clean rectangle images where h>w
    images = os.listdir(folder)
    print("Checking for sizes ...")
    for im in images:
        width, height = Image.open(os.path.join(folder,im)).size
        if height > width:
            print("Removing ", im, " size ", height, width)
            os.remove(os.path.join(folder, im))



def get_num_pixels(filepath='train_images'):
    """
    Check the number of pixels of images in a folder

    Parameters
    ----------
    filepath : str
        path to the file of images

    Returns
    -------
    str : 
        returns the dimensions of the images in a folder
    Examples
    --------
    >>> get_num_pixels('train_images')
    """
    images = os.listdir(filepath)
    for im in images:
        width, height = Image.open(os.path.join(filepath,im)).size
        print(im, width , "x", height)


def trim(image):
     """
    crop white spaces or borders of an image

    Parameters
    ----------
    img : image
        image object

    Returns
    -------
    image : 
        returns cropped image
    Examples
    --------
    >>> image_resize(img.jpg)
    """
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def image_resize(img, width, height):
    """
    resize pixels of image

    Parameters
    ----------
    img : image
        image object

    img : int
        desired height of image 

    img : int
        desired width of image    

    Returns
    -------
    str : 
        returns resized image
    Examples
    --------
    >>> image_resize(img.jpg, 5, 5)
    """
    return img.resize((width, height))


def rename_seq(path):
    """
    renaming images sequentially

    Parameters
    ----------
     path : str
        path to the file of imagese    

    Returns
    -------
    images : 
         returns renamed iamges
    Examples
    --------
    >>> rename_seq("combined")
    """
    
    images = os.listdir(path)
    for i, im in enumerate(images):
        src_path = os.path.join(path, im)
        dst_path = os.path.join(path, str(i)+'.jpg')
        os.rename(src_path,dst_path)


if __name__ == '__main__':
    app.run_server(debug=True)
