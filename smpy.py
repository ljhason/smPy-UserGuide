import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io
from matplotlib import patches
from matplotlib.ticker import MultipleLocator
import struct
import numpy as np
from skimage.feature import peak_local_max

def read_pma(pma_file_path):
    """
    Reads a .pma file and returns a list of frames as numpy arrays containing elements of type numpy.uint8.
    """
    try:
        with open(pma_file_path, "rb") as f:
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            f.seek(0, 2) 
            filesize = f.tell()  
            Nframes = (filesize - 4) // (X_pixels * Y_pixels)
            f.seek(4)  
            return [np.frombuffer(f.read(X_pixels*Y_pixels), dtype=np.uint8).reshape((Y_pixels, X_pixels)) for frame_idx in range(Nframes)]

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None

def avg_frame_arr(pma_file_path):
    """
    Reads a .pma file and returns the average frame as a numpy array.
    """
    try:

        Frames_data = read_pma(pma_file_path)
        avg_frame_data = np.mean(Frames_data, axis=0).astype(np.uint8)
        print(f"Sucessfully generated average frame")
        return avg_frame_data

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None

def find_peaks(image_path, min_distance, clip, block_size, scaler_percent):
    """
    From a grayscale image, finds all peaks bright spots in the image using a local maximum filter.
    """
    
    image = io.imread(image_path, as_gray=True).astype(np.uint8)
    std_dev = np.std(image)
    height, width = image.shape
    image_1 = image.copy()
    min_intensity = np.min(image_1)
    max_intensity = np.max(image_1)
    threshold = min_intensity + (scaler_percent / 100.0) * (max_intensity - min_intensity)
        
    background = np.zeros((height, width), dtype=np.float32)

    for i in range(8, height, block_size):
        for j in range(8, width, block_size):
            background[(i-8)//block_size, (j-8)//block_size] = np.min(image_1[i-8:i+8, j-8:j+8])
        

    background = np.clip(background.astype(np.uint8) - clip, 0, 255)
    image_1 = image - background
    image_2 = image_1.copy()
    med = np.median(image_1)
    image_2[image_2 < (med + 3*std_dev)] = 0
    peak_coords = peak_local_max(image_2, min_distance=int(min_distance), threshold_abs=threshold)
    
    return peak_coords, image_2

def shift_peaks(peaks, shift=[0, 256]):
    """
    Shifts the peaks coordinates by a specified amount.
    """
    return np.add(peaks, shift)


def draw_circle(radius, y_centre, x_centre, background_dim, colour = [255, 255, 0]):
    """"
    Creates an array of the same size as the background_dim and draws cicles of a specified radius centred on coordinates (y_centre, x_centre).
    """
    circle_array = np.zeros((background_dim, background_dim, 3), dtype=np.uint8)

    y = radius
    x = 0
    p = 1 - radius
    
    while y >= x:
        circle_array[y_centre + y, x_centre + x] = colour
        circle_array[y_centre - y, x_centre + x] = colour
        circle_array[y_centre + y, x_centre - x] = colour
        circle_array[y_centre - y, x_centre - x] = colour
        circle_array[y_centre + x, x_centre + y] = colour
        circle_array[y_centre - x, x_centre + y] = colour
        circle_array[y_centre + x, x_centre - y] = colour
        circle_array[y_centre - x, x_centre - y] = colour
         
        x += 1
        if p <= 0:
            p = p + 2 * x + 1
        else:
            y -= 1
            p = p + 2 * x - 2 * y + 1
    
    return circle_array

def count_circle(radius, y_centre=12, x_centre=12):
    """
    Counts the number of pixels in a circle of a specified radius centred on coordinates (y_centre, x_centre).
    """
    total = 0
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total


def SG_background_subtraction(pma_file_path, avg_frame_array, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    """
    Performs background subtraction using a static global calculation. Returns an array of corrected frames as numpy arrays with the same shape as the input array.
    """
    frames_data = read_pma(pma_file_path) 
    height, width = avg_frame_array.shape[:2]
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    filled_circle_mask = np.zeros((height, width), dtype=bool)

    if not CH_consideration:
        all_peaks_intensity = 0
        total_intensity = np.sum(avg_frame_array[:, :])
        num_of_peaks = len(y_centre_arr)
        num_of_frame_pixels = height * width

        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask
        
        all_peaks_intensity += np.sum(avg_frame_array[filled_circle_mask, 2])
        intensity_to_remove = ((total_intensity-all_peaks_intensity) // (num_of_frame_pixels-num_of_peak_pixels))
        for frame in frames_data: 
            frame = frame.astype(np.int16)
            frame = np.clip(frame - intensity_to_remove, 0, 255).astype(np.uint8)
            corrected_frames_data.append(frame)

    else: 
        all_peaks_intensity_CH1 = 0
        all_peaks_intensity_CH2 = 0

        total_intensity_CH1 = np.sum(avg_frame_array[:, :width//2])
        total_intensity_CH2 = np.sum(avg_frame_array[:, width//2:])
        num_of_peaks = len(y_centre_arr)//2 
        num_of_frame_pixels = height*width//2

        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        corrected_frames_data = []

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask|= mask

  
        filled_circle_mask_CH1 = filled_circle_mask[:, :width//2]
        filled_circle_mask_CH2 = filled_circle_mask[:, width//2:]
            
        all_peaks_intensity_CH1 += np.sum(avg_frame_array[:, :width//2][filled_circle_mask_CH1])
        all_peaks_intensity_CH2 += np.sum(avg_frame_array[:, width//2:][filled_circle_mask_CH2])

        intensity_to_remove_CH1 = ((total_intensity_CH1-all_peaks_intensity_CH1) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
        intensity_to_remove_CH2 = ((total_intensity_CH2-all_peaks_intensity_CH2) // (num_of_frame_pixels-num_of_peak_pixels)).astype(np.int16)
    
        for frame in frames_data: 
            frame = frame.astype(np.int16)
            frame_CH1 = np.clip(frame[:,:width//2] - intensity_to_remove_CH1, 0, 255).astype(np.uint8)
            frame_CH2 = np.clip(frame[:,width//2:] - intensity_to_remove_CH2, 0, 255).astype(np.uint8)
            frame = np.concatenate((frame_CH1, frame_CH2), axis=1)
            corrected_frames_data.append(frame)

    return corrected_frames_data

def DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=False):
    """
    Performs background subtraction using a dynamic global calculation. Returns an array of corrected frames as numpy arrays with the same shape as the input array.
    """

    frames_data = read_pma(pma_file_path)
    height, width = frames_data[0].shape[:2]

    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    filled_circle_mask = np.zeros((height, width), dtype=bool)
    corrected_frames_data = []

    if not CH_consideration:
        num_of_peaks = len(y_centre_arr)
        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        num_of_frame_pixels = height * width

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask |= mask
        
        for frame in frames_data:  
            all_peaks_intensity = np.sum(frame[filled_circle_mask])
            total_intensity = np.sum(frame)
            intensity_to_remove = np.int16((total_intensity - all_peaks_intensity) // (num_of_frame_pixels - num_of_peak_pixels))
            frame = frame.astype(np.int16) 
            frame -= intensity_to_remove 
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            corrected_frames_data.append(frame)

    else: 
        num_of_peaks = len(y_centre_arr)//2 
        num_of_peak_pixels = count_circle(radius) * num_of_peaks
        num_of_frame_pixels = frames_data[0].shape[0] * frames_data[0].shape[1]//2

        for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
            mask = (x_indices - x_centre) ** 2 + (y_indices - y_centre) ** 2 < radius ** 2
            filled_circle_mask |= mask
        
        filled_circle_mask_CH1 = filled_circle_mask[:, :width//2]
        filled_circle_mask_CH2 = filled_circle_mask[:, width//2:]
        
        for frame in frames_data:
            all_peaks_intensity_CH1= np.sum(frame[:, :width//2][filled_circle_mask_CH1])
            all_peaks_intensity_CH2= np.sum(frame[:, width//2:][filled_circle_mask_CH2])
            total_intensity_CH1 = np.sum(frame[:, : width//2])
            total_intensity_CH2 = np.sum(frame[:, width//2:])
            intensity_to_remove_CH1 = np.int16((total_intensity_CH1 - all_peaks_intensity_CH1) // (num_of_frame_pixels - num_of_peak_pixels)).astype(np.int16)
            intensity_to_remove_CH2 = np.int16((total_intensity_CH2 - all_peaks_intensity_CH2) // (num_of_frame_pixels - num_of_peak_pixels)).astype(np.int16)
            frame.astype(np.int16)
            frame_CH1 = np.clip(frame[:,:width//2] - intensity_to_remove_CH1, 0, 255).astype(np.uint8)
            frame_CH2 = np.clip(frame[:,width//2:] - intensity_to_remove_CH2, 0, 255).astype(np.uint8)
            frame = np.concatenate((frame_CH1, frame_CH2), axis=1)
            corrected_frames_data.append(frame)
    return corrected_frames_data

def update_annot(ind, scatter, peaks, label):
    """ 
    Updates the annotation position and text. 
    """
    idx = ind["ind"][0]
    y, x = peaks[idx]
    annot.xy = (scatter.get_offsets()[idx][0], scatter.get_offsets()[idx][1])
    annot.set_text(f"{label} Peak {idx}: (y, x) = ({y}, {x})")
    annot.set_visible(True)
    

def intensity_in_circle(input_array, radius, y_centre, x_centre):
    """
    From a 3D array, calculates the total intensity in a circle of a specified radius centred on coordinates (y_centre, x_centre).
    """
    total_intensity = 0
    intensity_arr = []

    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                intensity_arr.append(int(input_array[j][i][2]))
                total_intensity += int(input_array[j][i][2])

    return total_intensity, intensity_arr


def calc_FRET(I_D_list, I_A_list):
    """
    Calculates the FRET efficiency from the donor and acceptor intensities.
    """
    I_D, I_A = np.array(I_D_list), np.array(I_A_list)
    FRET_arr = I_A/(I_D + I_A)
    return FRET_arr.tolist()

def calc_distance(FRET_list, R_0):
    """
    Calculates the distance from the FRET efficiency using the Forster equation.
    """
    d = R_0 * ((1/np.array(FRET_list)) - 1)**(1/6)
    return d.tolist()

def generate_video(pma_file_path, fps=5):
    """
    Reads a .pma file and saves each frame as a .png image in a new directory.
    """
    try:
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        video_name = f"{output_name}.mp4"
        video_path= os.path.join(f"{output_name}_Video", video_name)
        
        if not os.path.exists(f"{output_name}_Video"):
            os.makedirs(f"{output_name}_Video")

        Frames_data = read_pma(pma_file_path)
        if not Frames_data or len(Frames_data) == 0:
            raise ValueError("No frames found in pma file.")
        
        height, width = Frames_data[0].shape[:2]
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        if not video.isOpened():
            raise IOError("Failed to open VideoWriter. Codec may not be supported.")
        
        for idx, frame in enumerate(Frames_data):
            if frame.dtype!= np.uint8:
                raise ValueError(f"Frame {idx} is not of type np.uint8.")
            
            video.write(frame)
        video.release()
        cv2.destroyAllWindows()

        print(f"Video sucessfully generated and saved: {video_path}")
        print(f"Frame Count: {(len(Frames_data))}")
    
    except Exception as e:
        print(f"Error generating video: {e}")
        return None


def avg_frame_png(pma_file_path):
    """
    Reads a .pma file and saves the average frame as a .png image in a new directory.
    """
    try:
        avg_frame_data = avg_frame_arr(pma_file_path)
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        image_file_name = f'{output_name}_Avg_Frame.png'
        if not os.path.exists(f"{output_name}_Avg_Frame"):
            os.makedirs(f"{output_name}_Avg_Frame")
        else:
            pass
        image = Image.fromarray(avg_frame_data)
        image.save(f"{output_name}_Avg_Frame/{image_file_name}")
        print(f"Average frame saved as: {image_file_name}")

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None

def dim_to_3(image):
    """
    Converts a grayscale image to a 3D RGB image.
    """
    return np.stack((image,) * 3, axis=-1)

def find_good_peaks(image_path, min_distance = 1, clip=10, block_size = 32, scaler_percent = 20, border=10, max_rad=3):
    """
    From a grayscale image, finds all peaks bright spots in the image using a local maximum filter and filters out the peaks that are too close to the edges and whose radius is larger than max_rad.
    """
    peaks_coords, image_2 = find_peaks(image_path, min_distance, clip, block_size, scaler_percent)
    bad_peaks = []
    good_peaks = []
    height, width = io.imread(image_path).shape

    for peak in peaks_coords:
        y, x = peak
        if y < border or y > height - border or x < border or x > width - border:
            bad_peaks.append(peak)
        elif image_2[y, x + max_rad+1] > 0 or image_2[y, x - max_rad] > 0 or image_2[y+max_rad+1, x ] > 0 or image_2[y-max_rad, x] > 0:
            bad_peaks.append(peak)
        else:
            good_peaks.append(peak)

    good_peaks = np.array(good_peaks)
    bad_peaks = np.array(bad_peaks)
    
    return good_peaks, bad_peaks

def find_pairs(peaks_1, peaks_2, tolerance=1, Channel_count=2, shift=[0,0]):
    """
    Finds pairs of peaks between two sets of peaks based on a specified tolerance and then unshifts the coordinates of the second set of peaks.
    """
    gp1_list = [tuple(peak) for peak in peaks_1]
    gp2_list = [tuple(peak) for peak in peaks_2]
    gp2_set = set(gp2_list)
    pair_arr_CH1 = []
    pair_arr_CH2 = []
    try: 
        if Channel_count == 2:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (256-tolerance <= abs(coord[1] - c[1]) <= 256+tolerance) and c not in pair_arr_CH2:
                            pair_arr_CH1.append(coord)
                            pair_arr_CH2.append(c)
                            break
        elif Channel_count == 1:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (abs(coord[1] - c[1]) <= tolerance) and c not in pair_arr_CH2:
                            pair_arr_CH1.append(coord)
                            pair_arr_CH2.append(c)
                            break
        else:
            print("Invalid Channel Count, please choose 1 or 2")
            return None

        if shift == [0, 0]:
            out_pair_arr_CH1 = np.array(pair_arr_CH1)
            out_pair_arr_CH2 = np.array(pair_arr_CH2)

        else:

            pair_arr_CH1 = np.array(pair_arr_CH1)
            pair_arr_CH2 = shift_peaks(np.array(pair_arr_CH2), shift = [-shift[0], -shift[1]])
            out_pair_arr_CH2 = pair_arr_CH2[(pair_arr_CH2[:,1] <= 502) & (pair_arr_CH2[:,1] >= 266) & (pair_arr_CH2[:, 0] <= 502) & (pair_arr_CH2[:, 0] >= 10)]
            out_pair_arr_CH1 = pair_arr_CH1[(pair_arr_CH2[:,1] <= 502) & (pair_arr_CH2[:,1] >= 266) & (pair_arr_CH2[:, 0] <= 502) & (pair_arr_CH2[:, 0] >= 10)]

    except Exception as e:
        print(f"Error finding pairs: {e}")
        return None

    return len(out_pair_arr_CH1), out_pair_arr_CH1, out_pair_arr_CH2


def find_polyfit_params(peaks_1, peaks_2, degree=2):
    """
    From two sets of peaks, finds the polynomial coefficients for the transformation between the two sets of peaks and returns the polynomial coefficients to a designated degree. 
    """
    y1, x1 = peaks_1[:, 0], peaks_1[:, 1] 
    y2, x2 = peaks_2[:, 0], peaks_2[:, 1] 

    params_x = np.polyfit(x1, x2, degree)
    params_y = np.polyfit(y1, y2, degree) 

    return params_x, params_y 

def apply_polyfit_params(CH1_peaks, params_x, params_y):
    """
    Applies the polynomial coefficients to the donor peaks and returns a numpy array of the mapped peak coordinates.
    """
    y1, x1 = CH1_peaks[:, 0], CH1_peaks[:, 1]
    x_mapped = np.polyval(params_x, x1)  
    y_mapped = np.polyval(params_y, y1) 
    return np.column_stack((y_mapped, x_mapped)) 


def plot_circle(image, y_centre, x_centre, radius=4, colour = [255, 255, 0]):
    """
    Plots a circle on the image at the specified coordinates (y_centre, x_centre) with the specified radius and colour.
    """
    circle_array = draw_circle(radius, y_centre, x_centre, image.shape[0])
    mask = (circle_array == [255, 255, 0]).all(axis=-1)
    try:
        if image.ndim == 2:
            image_3d = np.repeat(image[..., np.newaxis], 3, -1)
        elif image.ndim==3 and image.shape[2]==3:
            image_3d = image
    except Exception as e:
        print(f"Error plotting circle: {e}")
        return None

    image_3d[mask] = colour
    plt.imshow(image_3d)
    plt.show()


def init_annot(ax, text="", xy=(0, 0), xytext=(0, 10),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->", color="w")):
    """
    Initializes the annotation object.
    """
    global annot
    annot = ax.annotate(text, xy=xy, xytext=xytext, textcoords=textcoords, bbox=bbox, arrowprops=arrowprops)
    annot.set_visible(False)
    return annot


def print_coords_trigger(event, fig, scatter_data):
    """ 
    Checks if the mouse hovers over a peak and displays the coordinates and id of that peak. If clicked, prints the coordinates of the peak.
    """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            if event.name == "button_press_event":
        
                print(f"{label}_Peak{ind['ind'][0]} (y, x):({peaks[ind['ind'][0]][0]},{peaks[ind['ind'][0]][1]})")
            break

    annot.set_visible(visible)
    fig.canvas.draw_idle()

def interactive_zoom(event, fig, ax, scatter_data, image_3d, image_orig, zoom_size=5, CH1_zoom_axes=[0.75, 0.6, 0.2, 0.2], CH2_zoom_axes=[0.75, 0.3, 0.2, 0.2], CH1_colour = "orange", CH2_colour = "b"):
    """ 
    Checks if the mouse hovers over a point and updates annotation. If clicked, it displays the zoomed image of that peak and its corresponding peak in the other channel. 
    """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax: 
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1+1, x1_CH1:x2_CH1+1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2+1, x1_CH2:x2_CH2+1]

                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title("")
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH2})")
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1.5, edgecolor=CH1_colour, facecolor='none')
                ax.add_patch(rect1)
                ax_zoom_CH2.clear()
            
                
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1.5, edgecolor=CH2_colour, facecolor='none')
                ax.add_patch(rect2)
                

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def interactive_2CH_plots(event, pma_file_path, fig, ax, scatter_data, y_centre_arr, x_centre_arr, image_3d, image_orig, mask, radius=4, tpf=1/100, zoom_size=5, R_0=5.6, time_interval=1, background_treatment = "None", CH_consideration=False, time_unit="s", dist_unit="nm", Intense_axes_CH1=[0.48, 0.81, 0.5, 0.15], Intense_axes_CH2=[0.48, 0.56, 0.5, 0.15], FRET_axes=[0.48, 0.31, 0.5, 0.15], dist_axes=[0.48, 0.06, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.22, 0.06, 0.15, 0.15], CH1_colour = "orange", CH2_colour = "b", FRET_colour = "forestgreen", distance_colour = "palevioletred"):
    """ 
    Checks if the mouse hovers over a peak and updates annotation. If clicked, it displays the zoomed image and intensity time-series of that peak and its corresponding peak and the FRET and distance time series.
    """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                if background_treatment == "None":
                    Frames_data = read_pma(pma_file_path)
                elif background_treatment == "SG":
                    Frames_data = SG_background_subtraction(pma_file_path, image_3d, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                elif background_treatment == "DG":
                    Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)


                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                ax_intensity_CH1= fig.add_axes(Intense_axes_CH1)
                ax_intensity_CH2= fig.add_axes(Intense_axes_CH2)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1+1, x1_CH1:x2_CH1+1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2+1, x1_CH2:x2_CH2+1]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)):
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                
                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))

                ax_intensity_CH1.clear()
                ax_intensity_CH1.plot(time, tot_intensity_all_frames_CH1, color=CH1_colour, label='CH2')
                ax_intensity_CH1.set_title(f"Intensity v Time in Donor Peak {idx}, BT: {background_treatment}, CH consideration: {CH_consideration}")
                ax_intensity_CH1.set_xlabel(f'Time ({time_unit})')
                ax_intensity_CH1.set_ylabel('Intensity')
                ax_intensity_CH1.set_ylim(-255, max(tot_intensity_all_frames_CH1)+255)
                ax_intensity_CH1.set_xlim(0, time[-1])
                ax_intensity_CH1.grid()
                ax_intensity_CH1.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_intensity_CH1.yaxis.set_major_locator(MultipleLocator(500))

                ax_intensity_CH2.clear()
                ax_intensity_CH2.plot(time, tot_intensity_all_frames_CH2, color=CH2_colour, label='CH2')
                ax_intensity_CH2.set_title(f"Intensity v Time in Acceptor Peak {idx}, BT: {background_treatment}, CH consideration: {CH_consideration}")
                ax_intensity_CH2.set_xlabel(f'Time ({time_unit})')
                ax_intensity_CH2.set_ylabel('Intensity')
                ax_intensity_CH2.set_ylim(-255, max(tot_intensity_all_frames_CH2)+255)
                ax_intensity_CH2.set_xlim(0, time[-1])
                ax_intensity_CH2.grid()
                ax_intensity_CH2.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_intensity_CH2.yaxis.set_major_locator(MultipleLocator(500))

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color=FRET_colour)
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel(f'Time ({time_unit})')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()
                ax_FRET.xaxis.set_major_locator(MultipleLocator(time_interval))


                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color=distance_colour)
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel(f'Time ({time_unit})')
                ax_dist.set_ylabel(f'Distance ({dist_unit})')
                ax_dist.set_xlim(0, time[-1])
                ax_dist.grid()
                ax_dist.xaxis.set_major_locator(MultipleLocator(time_interval))


                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1.5, edgecolor=CH1_colour, facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1.5, edgecolor=CH2_colour, facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def interactive_2CH_plots_merged(event, pma_file_path, fig, ax, scatter_data, y_centre_arr, x_centre_arr, image_3d, image_orig, mask, radius=4, tpf=1/100, R_0=5.6, time_interval=1, zoom_size=5, background_treatment = "None", CH_consideration=False,time_unit="s", dist_unit="nm", Intense_axes=[0.48, 0.6, 0.5, 0.3], FRET_axes=[0.48, 0.35, 0.5, 0.15], dist_axes=[0.48, 0.1, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.23, 0.06, 0.15, 0.15], CH1_colour = "orange", CH2_colour = "b", FRET_colour = "forestgreen", distance_colour = "palevioletred"):
    """ 
    Checks if the mouse hovers over a peak and updates annotation. If clicked, it displays the zoomed image and intensity time-series of that peak and its corresponding peak and the FRET and distance time series.
    The intensity time series of the peaks are plotted in the same axes.
    """
    visible = False
    
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                if background_treatment == "None":
                    Frames_data = read_pma(pma_file_path)
                elif background_treatment == "SG":
                    Frames_data = SG_background_subtraction(pma_file_path, image_3d, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                elif background_treatment == "DG":
                    Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                else:
                    Frames_data = read_pma(pma_file_path)

                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)

                ax_intensity= fig.add_axes(Intense_axes)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1+1, x1_CH1:x2_CH1+1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2+1, x1_CH2:x2_CH2+1]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)): 
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)

                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))
                ax_intensity.clear()
                ax_intensity.plot(time, tot_intensity_all_frames_CH1, color=CH1_colour, label='CH1')
                ax_intensity.plot(time, tot_intensity_all_frames_CH2, color=CH2_colour, label='CH2')
                ax_intensity.set_title(f"Intensity v Time in Peak {idx}, BT: {background_treatment}, CH consideration: {CH_consideration}")
                ax_intensity.set_xlabel(f'Time ({time_unit})')
                ax_intensity.set_ylabel('Intensity')
                ax_intensity.set_ylim(-255, max(max(tot_intensity_all_frames_CH1), max(tot_intensity_all_frames_CH2))+255)
                ax_intensity.legend(bbox_to_anchor=(1.0, 1.22), loc='upper right')
                ax_intensity.grid()
                ax_intensity.set_xlim(0, time[-1])
                ax_intensity.xaxis.set_major_locator(MultipleLocator(time_interval)) 
                ax_intensity.yaxis.set_major_locator(MultipleLocator(500))  

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color=FRET_colour)
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel(f'Time ({time_unit})')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()
                ax_FRET.xaxis.set_major_locator(MultipleLocator(time_interval)) 


                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color=distance_colour)
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel(f'Time ({time_unit})')
                ax_dist.set_ylabel(f'Distance ({dist_unit})')
                ax_dist.set_xlim(0, time[-1])
                ax_dist.grid()
                ax_dist.xaxis.set_major_locator(MultipleLocator(time_interval))
            
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1.5, edgecolor=CH1_colour, facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1.5, edgecolor=CH2_colour, facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()

def display_time_series(pma_file_path, avg_image, peak_idx, CH1_arr, CH2_arr, tpf = 1/100, R_0=5.6, radius=4, time_interval=1, background_treatment = "None", CH_consideration=False, time_unit="s", dist_unit="nm", CH1_intensity_interval=500, CH2_intensity_interval=500, figsize=(15, 8), CH1_colour="orange", CH2_colour="b", FRET_colour="forestgreen", distance_colour="palevioletred"):
    """
    From a pma file, displays the time series of the total intensity in a circle of a specified radius centred on coordinates (y_centre, x_centre) for both channels.
    """
    y_CH1, x_CH1 = CH1_arr[peak_idx]
    y_CH2, x_CH2 = CH2_arr[peak_idx]
    tot_intensity_all_frames_CH1 = []
    tot_intensity_all_frames_CH2 = []
    y_centre_arr = np.concatenate((CH1_arr[:, 0], CH2_arr[:, 0]))
    x_centre_arr = np.concatenate((CH1_arr[:, 1], CH2_arr[:, 1]))
    if background_treatment == "None":
        Frames_data = read_pma(pma_file_path)
    elif background_treatment == "SG":
        Frames_data = SG_background_subtraction(pma_file_path, avg_image, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
    elif background_treatment == "DG":
        Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
    else:
        Frames_data = read_pma(pma_file_path)


    circle_array_new = draw_circle(4, y_centre_arr, x_centre_arr, avg_image.shape[0])
    mask = (circle_array_new == [255, 255, 0]).all(axis=-1)


    for i in range(len(Frames_data)): 
        if Frames_data[i].ndim == 2:
            frame_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
        elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
            frame_3d = Frames_data[i]
        frame_3d[mask] = [255, 255, 0]

        total_intensity_CH1,_ = intensity_in_circle(frame_3d, radius, y_CH1, x_CH1)
        total_intensity_CH2,_ = intensity_in_circle(frame_3d, radius, y_CH2, x_CH2)
        tot_intensity_all_frames_CH1.append(total_intensity_CH1)
        tot_intensity_all_frames_CH2.append(total_intensity_CH2)

    time = np.linspace(0, (len(tot_intensity_all_frames_CH1)-1)*tpf, len(tot_intensity_all_frames_CH1))
    fig, ax = plt.subplots(4, 1, figsize=figsize)
    fig.subplots_adjust(hspace=1)
    ax[0].plot(time, tot_intensity_all_frames_CH1, color=CH1_colour)
    ax[0].set_title(f'Intensity v Time in Donor Peak {peak_idx}, BT: {background_treatment}, CH consideration: {CH_consideration}')
    ax[0].set_ylabel('Intensity')
    ax[0].set_xlabel(f'Time ({time_unit})')
    ax[0].set_ylim(-255, max(tot_intensity_all_frames_CH1)+255)
    ax[0].set_xlim(0, time[-1])
    ax[0].grid()
    ax[0].xaxis.set_major_locator(MultipleLocator(time_interval))
    ax[0].yaxis.set_major_locator(MultipleLocator(CH1_intensity_interval))  


    ax[1].plot(time, tot_intensity_all_frames_CH2, color=CH2_colour)
    ax[1].set_title(f'Intensity v Time in Acceptor Peak {peak_idx}, BT: {background_treatment}, CH consideration: {CH_consideration}')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel(f'Time ({time_unit})')
    ax[1].set_ylim(-255, max(tot_intensity_all_frames_CH2)+255)
    ax[1].grid()
    ax[1].set_xlim(0, time[-1])
    ax[1].xaxis.set_major_locator(MultipleLocator(time_interval))
    ax[1].yaxis.set_major_locator(MultipleLocator(CH2_intensity_interval))

    FRET_values = calc_FRET(np.array(tot_intensity_all_frames_CH1), np.array(tot_intensity_all_frames_CH2))
    ax[2].plot(time, FRET_values, color=FRET_colour)
    ax[2].set_title(f'FRET v Time (Peak {peak_idx})')
    ax[2].set_xlabel(f'Time ({time_unit})')
    ax[2].set_ylabel('FRET Efficiency')
    ax[2].grid()
    ax[2].set_xlim(0, time[-1])
    ax[2].xaxis.set_major_locator(MultipleLocator(time_interval)) 


    distance = calc_distance(FRET_values, R_0)
    ax[3].plot(time, distance, color=distance_colour)
    ax[3].set_title(f'Distance v Time (Peak {peak_idx})')
    ax[3].set_xlabel(f'Time ({time_unit})')
    ax[3].set_ylabel(f'Distance ({dist_unit})')
    ax[3].grid()
    ax[3].set_xlim(0, time[-1])
    ax[3].xaxis.set_major_locator(MultipleLocator(time_interval)) 
    plt.show()


def find_polyfit_params_3CH(peaks_1, peaks_2, peaks_3, degree=2):
    """
    Returns the polynomial coefficients for the transformation between the peaks of three channels, with a degree of 2.
    peaks_1 -> peaks_2 and peaks_1 -> peaks_3
    """
    y1, x1 = peaks_1[:, 0], peaks_1[:, 1] 
    y2, x2 = peaks_2[:, 0], peaks_2[:, 1] 
    y3, x3 = peaks_3[:, 0], peaks_3[:, 1]

    params_x_12 = np.polyfit(x1, x2, degree)
    params_y_12 = np.polyfit(y1, y2, degree)
    params_x_13 = np.polyfit(x1, x3, degree)
    params_y_13 = np.polyfit(y1, y3, degree)


    return params_x_12, params_y_12, params_x_13, params_y_13

def find_trip(peaks_1, mapped_CH2, mapped_CH3, tolerance=4, shift_CH2=[0,0], shift_CH3=[0,0]):
    """
    Finds the triplet of peaks that match in three channels based on the given tolerance and a given linear shift of CH2 and CH3 peaks (if needed).
    """
    matched_CH1 = []
    matched_CH2 = []
    matched_CH3 = []

    gp1_list = [tuple(peak) for peak in peaks_1]
    mapped_CH2_set = set([tuple(peak) for peak in mapped_CH2])
    mapped_CH3_set = set([tuple(peak) for peak in mapped_CH3])

    for ch1_peak in gp1_list:
        y1, x1 = ch1_peak

        ch2_match = None
        for ch2_peak in mapped_CH2_set:
            y2, x2 = ch2_peak
            if abs(y1 - y2) <= tolerance and abs(x2 - x1 - 171) <= tolerance:
                ch2_match = ch2_peak
                break

        ch3_match = None
        for ch3_peak in mapped_CH3_set:
            y3, x3 = ch3_peak
            if abs(y1 - y3) <= tolerance and abs(x3 - x1 - 342) <= tolerance:
                ch3_match = ch3_peak
                break
        if ch2_match is not None and ch3_match is not None:
            matched_CH1.append(ch1_peak)
            matched_CH2.append(ch2_match)
            matched_CH3.append(ch3_match)
    
    matched_CH1 = np.array(matched_CH1)
    matched_CH2 = shift_peaks(np.array(matched_CH2), shift=[-shift_CH2[0], -shift_CH2[1]])
    matched_CH3 = shift_peaks(np.array(matched_CH3), shift=[-shift_CH3[0], -shift_CH3[1]])
    out_pair_arr_CH2 = matched_CH2[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]
    out_pair_arr_CH1 = matched_CH1[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]
    out_pair_arr_CH3 = matched_CH3[(matched_CH3[:,1] <= 502) & (matched_CH3[:,1] >= 352) & (matched_CH3[:, 0] <= 502) & (matched_CH3[:, 0] >= 10) & (matched_CH2[:,1] <= 332) & (matched_CH2[:,1] >= 171) & (matched_CH2[:, 0] <= 502) & (matched_CH2[:, 0] >= 10)]

    return len(out_pair_arr_CH1), out_pair_arr_CH1, out_pair_arr_CH2, out_pair_arr_CH3


def interactive_3CH_plots(event, pma_file_path, fig, ax, scatter_data, y_centre_arr, x_centre_arr, image_3d, image_orig, mask, radius=4, tpf=1/100, R_0_1=5.6, R_0_2=5.6, time_interval=10, intensity_interval=500, FRET_interval=0.2, distance_1_interval=0.5, distance_2_interval=0.2, zoom_size=5, background_treatment="None", CH_consideration=False, time_unit="s", dist_unit="nm", Intense_axes=[0.48, 0.81, 0.5, 0.15], FRET_axes = [0.48, 0.56, 0.5, 0.15], distance_axes_1=[0.48, 0.31, 0.5, 0.15], distance_axes_2=[0.48, 0.06, 0.5, 0.15],  CH1_zoom_axes=[0.0005, 0.05, 0.15, 0.15], CH2_zoom_axes=[0.135, 0.05, 0.15, 0.15], CH3_zoom_axes=[0.27, 0.05, 0.15, 0.15], CH1_colour = "orange", CH2_colour = "b", CH3_colour = "darkviolet", FRET_1_colour = "orange", FRET_2_colour = "b", distance_1_colour = "orange", distance_2_colour = "b"):
    """
    From a pma file, displays the time series of the total intensity in a circle of a specified radius centred on coordinates (y_centre, x_centre) for all channels.
    """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                if background_treatment == "None":
                    Frames_data = read_pma(pma_file_path)
                elif background_treatment == "SG":
                    Frames_data = SG_background_subtraction(pma_file_path, image_3d, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                elif background_treatment == "DG":
                    Frames_data = DG_background_subtraction(pma_file_path, radius, y_centre_arr, x_centre_arr, CH_consideration=CH_consideration)
                else:
                    Frames_data = read_pma(pma_file_path)
                    
                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)
                ax_zoom_CH3 = fig.add_axes(CH3_zoom_axes)

                ax_intensity = fig.add_axes(Intense_axes)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_distance_1 = fig.add_axes(distance_axes_1)
                ax_distance_2 = fig.add_axes(distance_axes_2)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)
                y_CH3, x_CH3 = scatter_data[2][1][idx]
                x1_CH3, x2_CH3 = max(0, x_CH3 - zoom_size), min(image_3d.shape[1], x_CH3 + zoom_size)
                y1_CH3, y2_CH3 = max(0, y_CH3 - zoom_size), min(image_3d.shape[0], y_CH3 + zoom_size)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1+1, x1_CH1:x2_CH1+1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2+1, x1_CH2:x2_CH2+1]
                zoomed_image_CH3 = image_orig[y1_CH3:y2_CH3+1, x1_CH3:x2_CH3+1]

                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"CH1: ({y_CH1}, {x_CH2})")
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1.5, edgecolor=CH1_colour, facecolor='none')
                ax.add_patch(rect1)
                ax_zoom_CH2.clear()
            
                
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"CH2: ({y_CH2}, {x_CH2})")
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1.5, edgecolor=CH2_colour, facecolor='none')
                ax.add_patch(rect2)

                ax_zoom_CH3.clear()
                ax_zoom_CH3.imshow(zoomed_image_CH3, cmap="gray")
                ax_zoom_CH3.set_xticks([])
                ax_zoom_CH3.set_yticks([])
                ax_zoom_CH3.set_title(f"CH3: ({y_CH3}, {x_CH3})")
                rect3 = patches.Rectangle((x1_CH3, y1_CH3), x2_CH3 - x1_CH3, y2_CH3 - y1_CH3, linewidth=1.5, edgecolor=CH3_colour, facecolor='none')
                ax.add_patch(rect3)

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []
                tot_intensity_all_frames_CH3 = []

                for i in range(len(Frames_data)):
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    total_intensity_CH3,_ = intensity_in_circle(frame_data_3d, radius, y_CH3, x_CH3)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                    tot_intensity_all_frames_CH3.append(total_intensity_CH3)

                time = np.linspace(0, (len(tot_intensity_all_frames_CH1)-1) * tpf, len(tot_intensity_all_frames_CH1))

                ax_intensity.clear()
                ax_intensity.plot(time, tot_intensity_all_frames_CH1, color=CH1_colour, label='CH1')
                ax_intensity.plot(time, tot_intensity_all_frames_CH2, color=CH2_colour, label='CH2')
                ax_intensity.plot(time, tot_intensity_all_frames_CH3, color=CH3_colour, label='CH3')
                ax_intensity.set_title(f"Intensity v Time in Peak {idx}, BT: {background_treatment}, CH consideration: {CH_consideration}")
                ax_intensity.set_xlabel(f"Time ({time_unit})")
                ax_intensity.set_ylabel("Intensity")
                ax_intensity.set_xlim(0, time[-1])
                ax_intensity.grid()
                ax_intensity.legend(loc="upper right")
                ax_intensity.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_intensity.yaxis.set_major_locator(MultipleLocator(intensity_interval))

                FRET_values_1 = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                FRET_values_2 = calc_FRET(tot_intensity_all_frames_CH2, tot_intensity_all_frames_CH3)           
                ax_FRET.clear()
                ax_FRET.plot(time, FRET_values_1, color=FRET_1_colour, label='CH1-CH2')
                ax_FRET.plot(time, FRET_values_2, color=FRET_2_colour, label='CH2-CH3')
                ax_FRET.set_title(f"FRET v Time (Triplet {idx})")
                ax_FRET.set_xlabel(f"Time ({time_unit})")
                ax_FRET.set_ylabel("Intensity")
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()
                ax_FRET.legend(bbox_to_anchor=(1, 1.4),loc="upper right")

                ax_FRET.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_FRET.yaxis.set_major_locator(MultipleLocator(FRET_interval))

                distance_1 = calc_distance(FRET_values_1, R_0_1)

                ax_distance_1.clear()
                ax_distance_1.plot(time, distance_1, color=distance_1_colour)
                ax_distance_1.set_title(f"CH1-CH2 Distance v Time (Peak ID: {idx})")
                ax_distance_1.set_xlabel(f"Time ({time_unit})")
                ax_distance_1.set_ylabel("Distance (nm)")
                ax_distance_1.set_xlim(0, time[-1])
                ax_distance_1.grid()
                ax_distance_1.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_distance_1.yaxis.set_major_locator(MultipleLocator(distance_1_interval))
                
                distance_2 = calc_distance(FRET_values_2, R_0_2)
                ax_distance_2.clear()
                ax_distance_2.plot(time, distance_2, color=distance_2_colour)
                ax_distance_2.set_title(f"CH2-CH3 Distance v Time (Peak ID: {idx})")
                ax_distance_2.set_xlabel(f"Time ({time_unit})")
                ax_distance_2.set_ylabel(f"Distance ({dist_unit})")
                ax_distance_2.set_xlim(0, time[-1])
                ax_distance_2.grid()
                ax_distance_2.xaxis.set_major_locator(MultipleLocator(time_interval))
                ax_distance_2.yaxis.set_major_locator(MultipleLocator(distance_2_interval))

    annot.set_visible(visible)
    fig.canvas.draw_idle()


    