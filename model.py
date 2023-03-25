import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle5 as pickle

def get_contour(img):
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    # convert the image to grayscale format
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def calculate_color2(img_rgb): 
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    r_max = np.max(r)
    r_min = np.min(r)
    r_mean = np.mean(r)
    r_var = np.var(r)
    g_max = np.max(g)
    g_min = np.min(g)
    g_mean = np.mean(g)
    g_var = np.var(g)
    b_max = np.max(b)
    b_min = np.min(b)
    b_mean = np.mean(b)
    b_var = np.var(b)
    return r_max, r_min, r_mean, r_var, g_max, g_min, g_mean, g_var, b_max, b_min, b_mean, b_var
    
def calculate_symmetry(mask, show=False): 
    mask2 = np.fliplr(mask) 
    mask3 = np.flipud(mask) 
    A1 = mask | mask2
    O1 = mask & mask2
    A2 = mask | mask3
    O2 = mask & mask3
    P1 = np.sum(O1) / (np.sum(A1) + 0.0001) 
    P2 = np.sum(O2) / (np.sum(A2) + 0.0001) 

    if (show):
        fig1, axs1 = plt.subplots(1, 5, figsize=(10, 10))
        axs1 = axs1.ravel()
        axs1[0].imshow(mask,cmap='binary')
        axs1[0].set_title('original')
        axs1[1].imshow(mask2,cmap='binary')
        axs1[1].set_title('flip horizontally')
        axs1[2].imshow(mask3, cmap='binary')
        axs1[2].set_title('flip vertically')
        axs1[3].imshow(A1, cmap='binary')
        axs1[3].set_title('union mask')
        axs1[4].imshow(O1, cmap='binary')
        axs1[4].set_title('common mask')
    return (P1 + P2) / 2

def calculate_area(mask):
    return np.sum(mask)

def get_features(img_path):
    img = plt.imread(img_path)
    mask = get_contour(img)
    symm = calculate_symmetry(mask)
    area = calculate_area(mask)
    r_max, r_min, r_mean, r_var, g_max, g_min, g_mean, g_var, b_max, b_min, b_mean, b_var = calculate_color2(img)

    ans = np.array([[r_max, r_min, r_mean, r_var, g_max, g_min, g_mean, g_var, b_max, b_min, b_mean, b_var, symm, area]])
    df = pd.DataFrame(ans, columns=['r_max', 'r_min', 'r_mean', 'r_var', 'g_max', 'g_min', 'g_mean', 'g_var', 'b_max', 'b_min', 'b_mean', 'b_var', 'symm', 'area'])
    return df

def predict(img_path):
    # load saved model
    feats = get_features(img_path)
    r_max = str(feats['r_max'].values[0]) 
    r_min = str(feats['r_min'].values[0]) 
    r_mean = str(feats['r_mean'].values[0]) 
    r_var = str(feats['r_var'].values[0]) 
    g_max = str(feats['g_max'].values[0]) 
    g_min = str(feats['g_min'].values[0]) 
    g_mean = str(feats['g_mean'].values[0]) 
    g_var = str(feats['g_var'].values[0]) 
    b_max = str(feats['b_max'].values[0]) 
    b_min = str(feats['b_min'].values[0]) 
    b_mean = str(feats['b_mean'].values[0]) 
    b_var = str(feats['b_var'].values[0]) 
    symm = str(feats['symm'].values[0]) 
    area = str(feats['area'].values[0]) 
  
    with open('./static/models/model_pkl' , 'rb') as f:
        rand_forest = pickle.load(f)
        return rand_forest.predict(feats)[0], r_max, r_min, r_mean, r_var, g_max, g_min, g_mean, g_var, b_max, b_min, b_mean, b_var, symm, area

if __name__ == "__main__":
    print(predict('./static/uploads/ISIC_0000022.jpg'))
    print(predict('./static/uploads/ISIC_0000003.jpg'))
   