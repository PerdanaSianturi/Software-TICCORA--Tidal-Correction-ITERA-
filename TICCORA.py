import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.scrolledtext import ScrolledText

my = 6.673e-8 # Newton's gravitational constant in cgs units
M = 7.3537e25 # Mass of the moon in grams
S = 1.993e33  # Mass of the sun in grams
e = 0.054899720  # Eccentricity of the moon's orbit
m = 0.074804  # Ratio of mean motion of the sun to that of the moon
c = 3.84402e10 # Mean distance between the centers of the earth and the moon in cm
c1 = 1.495e13  # Mean distance between centers of the earth and sun in cm
a = 6.378270e8  # Earth's equatorial radius in cm
i = 0.08979719 # (i) Inclination of the moon's orbit to the ecliptic
h = 0.6  # vertical changes in the Earth's surface due to tides
k = 0.3  # changes in the earth's gravitational potential due to tides
LoveFactor = 1 + h - 1.5 * k # Love Number
omega = np.radians(23.452) # Inclination of the Earth's equator to the ecliptic
origin_date = datetime(1899, 12, 31, 12, 00, 00) # Noon Dec 31, 1899


def About():
    About_text = """
    TICCORA - TIDAL CORRECTION ITERA - Gravity Method
    
    Version 1.0
    
    Perdana Muliado Sianturi and Purwaditya Nugraha
    
    Email: perdanamsianturi@gmail.com
    
    
This application was made to fulfill the final project of Perdana Muliado Sianturi, S1 Geophysical Engineering study program at the Sumatra Institute of Technology.
                """

    messagebox.showinfo("About TICCORA", About_text)


def Reference():
    Reference_text = """
Longman, I.M., Journal of Geophysical Research, Volume 64, No.12. Formulas for Computing the Tidal Accelerations Due to the Moon and Sun, December 1959
                """
    messagebox.showinfo("Reference", Reference_text)


def calculate_tidal_correction(lat, lon, alt, start_time, end_time, utc_offset, time_interval):
    try:
        time_list = [start_time + timedelta(seconds=i) for i in range(0, int((end_time - start_time).total_seconds()) + 1, time_interval)]
        result_df = pd.DataFrame(columns=['Time', 'g0', 'gm', 'gs'])
        time_values, g0_values, gm_values, gs_values = [], [], [], []
        for time in time_list:
            g0, gm, gs = solve_longman_tide(lat, lon, alt, time - timedelta(hours=utc_offset))
            time_values.append(time)
            g0_values.append(g0)
            gm_values.append(gm)
            gs_values.append(gs)
        result_df['Time'] = time_values
        result_df['g0'] = g0_values
        result_df['gm'] = gm_values
        result_df['gs'] = gs_values
        return result_df
    except Exception as e:
        print(f"Error: {e}")
        return None


def convert_time_interval(interval, unit):
    if unit == "Second":
        return interval
    elif unit == "Minute":
        return interval * 60
    elif unit == "Hour":
        return interval * 3600


def Apply():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        alt = float(alt_entry.get())
        start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
        utc_offset = int(utc_offset_combobox.get()) 
        time_interval = int(time_interval_entry.get()) 
        time_interval_unit = time_interval_combobox.get()
        time_interval_seconds = convert_time_interval(time_interval, time_interval_unit)

        global result_df
        result_df = calculate_tidal_correction(lat, lon, alt, start_time, end_time, utc_offset, time_interval_seconds)
        if result_df is not None:
            result_text.configure(state="normal")
            result_text.delete(1.0, tk.END)
            header = "        Time         |     g0(mGal)   |     gm(mGal)    |    gs(mGal)   \n"
            divider = "=" * 72 + "\n"
            result_text.insert(tk.END, header)
            result_text.insert(tk.END, divider)
            formatted_result = format_dataframe(result_df)
            result_text.insert(tk.END, formatted_result)
            result_text.configure(state='disabled')
            
            plot_results(result_df)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def plot_results(result_df):
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['Time'], result_df['g0'], '*c', label='g0')
    plt.plot(result_df['Time'], result_df['gm'], '.', label='gm (Moon Tidal)')
    plt.plot(result_df['Time'], result_df['gs'], '+', label='gs (Sun Tidal)')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (mGal)')
    plt.title('Tidal Correction')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_julian_century(dates: np.ndarray, utc_offset: int = 0):
    delta = dates - origin_date
    days = delta.days + delta.seconds / 3600. / 24.
    t0 = dates.hour + dates.minute / 60. + dates.second / 3600.
    t0 -= utc_offset
    t0 = (t0 + 24) % 24
    return days / 36525, t0


def solve_longman_tide(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, time: np.ndarray):
    T, t0 = calculate_julian_century(time)
    T2 = T ** 2
    T3 = T ** 3

    L = -lon
    lamda = np.radians(lat)
    cos_lamda = np.cos(lamda)
    sin_lamda = np.sin(lamda)
    H = alt * 100
    s = 4.720023438 + 8399.7093 * T + 4.40695e-05 * T2 + 3.29e-08 * T3
    p = 5.835124721 + 71.018009 * T + 1.80546e-04 * T2 + 2.181e-07 * T3
    h = 4.881627934 + 628.33195 * T + 5.2796e-06 * T2
    N = 4.523588570 - 33.757153 * T + 3.67488e-05 * T2 + 3.87e-08 * T3
    cosN = np.cos(N)
    sinN = np.sin(N)
    I = np.arccos(np.cos(omega) * np.cos(i) - np.sin(omega) * np.sin(i) * cosN)
    v = np.arcsin(np.sin(i) * sinN / np.sin(I))
    t = np.radians(15. * (t0 - 12) - L)
    chi = t + h - v
    cos_alpha = cosN * np.cos(v) + sinN * np.sin(v) * np.cos(omega)
    sin_alpha = np.sin(omega) * sinN / np.sin(I)
    ny = 2 * np.arctan(sin_alpha / (1 + cos_alpha))
    ksi = s- (N - ny)
    l = ksi + 2 * e * np.sin(s - p) + (5. / 4) * e * e * np.sin(2 * (s - p)) + (15. / 4) * m * e * np.sin(s - 2 * h + p) + (11. / 8) * m * m * np.sin(2 * (s - h))
    p1 = 4.908229467 + 3.0005264e-02 * T + 7.9024e-06 * T2 + 5.81e-08 * T3
    e1 = 0.0168 - 4.2e-05 * T - 1.26e-07 * T2
    chi1 = t + h
    l1 = h + 2 * e1 * np.sin(h - p1)
    cos_theta = sin_lamda * np.sin(I) * np.sin(l) + cos_lamda * (np.cos(0.5 * I) ** 2 * np.cos(l - chi) + np.sin(0.5 * I) ** 2 * np.cos(l + chi))
    cos_phi = sin_lamda * np.sin(omega) * np.sin(l1) + cos_lamda * (np.cos(0.5 * omega) ** 2 * np.cos(l1 - chi1) + np.sin(0.5 * omega) ** 2 * np.cos(l1 + chi1))

    C = np.sqrt(1. / (1 + 0.006738 * sin_lamda ** 2))
    r = C * a + H
    aprime = 1. / (c * (1 - e * e))
    aprime1 = 1. / (c1 * (1 - e1 * e1))
    d = 1. / ((1. / c) + aprime * e * np.cos(s - p) + aprime * e ** 2 * np.cos(2 * (s - p)) + (15. / 8) * aprime * m * e * np.cos(s - 2 * h + p) + aprime * m * m * np.cos(2 * (s - h)))
    D = 1. / ((1. / c1) + aprime1 * e1 * np.cos(h - p1))
    gm = (my * M * r / d ** 3) * (3 * cos_theta ** 2 - 1) + (1.5 * (my * M * r ** 2 / d ** 4) * (5 * cos_theta ** 3 - 3 * cos_theta))
    gs = my * S * r / D ** 3 * (3 * cos_phi ** 2 - 1)
    g0 = (gm + gs) * 1e3 * LoveFactor

    return g0, gm * 1e3, gs * 1e3


def search(event=None):
    search_query = search_entry.get().strip()
    if search_query:
        try:
            search_time = datetime.strptime(search_query, "%Y-%m-%d %H:%M:%S")
            result = result_df[result_df['Time'] == search_time]
            if not result.empty:
                result_text.configure(state='normal')
                result_text.delete(1.0, tk.END)
                header = "        Time         |     g0(mGal)   |     gm(mGal)    |    gs(mGal)   \n"
                divider = "=" * 72 + "\n"
                result_text.insert(tk.END, header)
                result_text.insert(tk.END, divider)
                formatted_result = format_dataframe(result)
                result_text.insert(tk.END, formatted_result) 
                result_text.configure(state='disabled')
            else:
                messagebox.showinfo("Search Result", "No matching data found for the given time.")
        except ValueError:
            messagebox.showerror("Invalid Format", "Please enter the time in the format: YYYY-MM-DD HH:MM:SS")

    search_entry.bind("<KeyRelease>", search)

def Export():
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            result_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", "Data has been exported to CSV successfully.")
    except Exception as e:
        messagebox.showerror("Export Error", f"An error occurred while exporting data: {e}")


def format_dataframe(df):
    formatted_output = ""
    for index, row in df.iterrows():
        formatted_output += f"{row['Time'].strftime('%Y-%m-%d %H:%M:%S'):^20} | {row['g0']:^10.12f} | {row['gm']:^10.12f} | {row['gs']:^10.12f}\n"
    return formatted_output

root = tk.Tk()
root.title("TICCORA - Tidal Correction ITERA")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.N, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

lat_label = ttk.Label(mainframe, text="Latitude                                                                 =")
lat_label.grid(column=1, row=1, sticky=tk.W)
lat_entry = ttk.Entry(mainframe)
lat_entry.grid(column=2, row=1)

lon_label = ttk.Label(mainframe, text="Longitude                                                             =")
lon_label.grid(column=1, row=2, sticky=tk.W)
lon_entry = ttk.Entry(mainframe)
lon_entry.grid(column=2, row=2)

alt_label = ttk.Label(mainframe, text="Altitude                                                                 =")
alt_label.grid(column=1, row=3, sticky=tk.W)
alt_entry = ttk.Entry(mainframe)
alt_entry.grid(column=2, row=3)

start_time_label = ttk.Label(mainframe, text="Start Time (YYYY-MM-DD hh:mm:ss)              =")
start_time_label.grid(column=1, row=4, sticky=tk.W)
start_time_entry = ttk.Entry(mainframe)
start_time_entry.grid(column=2, row=4)

end_time_label = ttk.Label(mainframe, text="End Time (YYYY-MM-DD hh:mm:ss)                =")
end_time_label.grid(column=1, row=5, sticky=tk.W)
end_time_entry = ttk.Entry(mainframe)
end_time_entry.grid(column=2, row=5)

utc_offset_label = ttk.Label(mainframe, text="UTC (Coordinated Universal Time)                    =")
utc_offset_label.grid(column=1, row=6, sticky=tk.W)
utc_offset_combobox = ttk.Combobox(mainframe, values=[str(i) for i in range(-12, 13)], state="readonly")
utc_offset_combobox.grid(column=2, row=6)

time_interval_label = ttk.Label(mainframe, text="Time Interval (Second/Minute/Hour)               =")
time_interval_label.grid(column=1, row=7, sticky=tk.W)
time_interval_entry = ttk.Entry(mainframe)
time_interval_entry.grid(column=2, row=7)
time_interval_combobox = ttk.Combobox(mainframe, values=["Second", "Minute", "Hour"], state="readonly")
time_interval_combobox.grid(column=3, row=7)
time_interval_combobox.current(0)

submit_button = ttk.Button(mainframe, text="Apply", command=Apply)
submit_button.grid(column=2, row=8, sticky=tk.W)

export_button = ttk.Button(mainframe, text="Reference", command=Reference)
export_button.grid(column=3, row=8, sticky=tk.W)

reference_button = ttk.Button(mainframe, text="Export", command=Export)
reference_button.grid(column=2, row=9, sticky=tk.W)

About_button = ttk.Button(mainframe, text="About", command=About)
About_button.grid(column=3, row=9, sticky=tk.W)

search_label = ttk.Label(mainframe, text="Search                                                                    =")
search_label.grid(column=1, row=10, sticky=tk.W)
search_entry = ttk.Entry(mainframe)
search_entry.grid(column=2, row=10, sticky=(tk.W, tk.E))
search_entry.bind("<KeyRelease>", search)

result_text = ScrolledText(mainframe, height=25, width=100)
result_text.grid(column=1, row=11, columnspan=3, sticky=(tk.W, tk.E))

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
