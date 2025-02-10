import tkinter as tk

import scipy
import Pre_processing as pp
from tkinter import filedialog, messagebox
import numpy as np
import scipy.stats as stats

# 定義全域變數
tmplistTemp = []

# 選取檔案的函式
def browse_file():
    filename = filedialog.askopenfilename(title="Select a File")
    # 如果選取檔案，則將檔案名稱顯示在畫面上
    if filename:
        file_label.config(text=filename)
    #先將檔案中的所有數字讀取出來，並顯示統計數據
    tmplistTemp = pp.read_ALL_from_file(filename)
    #將listTemp轉換為np.array
    tmplistTemp = np.array(tmplistTemp)
    
    #tmplistTemp.showinfo顯示出來,數字顯示到小數點第二位
    # messagebox.showinfo("檔案內容:", "最大值:"
    # + str(round(np.max(tmplistTemp[:,0]),2)) + "\n"
    # + "最小值:" + str(round(np.min(tmplistTemp[:,0]),2)) + "\n"
    # + "平均值:" + str(round(np.mean(tmplistTemp[:,0]),2)) + "\n"
    # + "標準差:" + str(round(np.std(tmplistTemp[:,0]),2)) + "\n"
    # + "中位數:" + str(round(np.median(tmplistTemp[:,0]),2)) + "\n"
    # + "總共幾個點:" + str(len(tmplistTemp[:,0]))
    # )
    #tmplistTemp
    # pp.draw_hist(tmplistTemp[:,0])
    #tmplistTemp,第一個參數是時間序列，第二個參數是數據
    #pp.drow_time_series(tmplistTemp[:,0],tmplistTemp[:,2])
    #將listTemp的數據中最穩定的那段區間的value顯示出來，並將回傳的區間最大值放到畫面上的區間_from_entry和區間_to_entry
    _max, _min = pp.draw_stable_section(tmplistTemp[:,0],tmplistTemp[:,2])
    # 將最大值和最小值轉換為整數，去除小數點
    _max = int(_max)+1
    _min = int(_min)
    #先將區間_from_entry和區間_to_entry清空
    區間_from_entry.delete(0, tk.END)
    區間_to_entry.delete(0, tk.END)
    #將區間_from_entry和區間_to_entry填入最大值和最小值
    區間_from_entry.insert(0,str(_min))
    區間_to_entry.insert(0,str( _max))
    #listTemp = pp.draw_stable_section(tmplistTemp[:,0],tmplistTemp[:,2])

    #將listTemp轉換為np.array
    #listTemp = np.array(listTemp)

    #將listTemp的數據畫成時序圖,第一個參數是時間序列，第二個參數是數據
    # pp.drow_time_series(listTemp[:,0],listTemp[:,1])

    #pp.draw_time_series(listTemp[:,0])
    #將listTemp的數據畫成散點圖
    # pp.draw_scatter(listTemp[:,0])
    
    return tmplistTemp

#Teager能量算子(TEO)的函數
def Teager_power_function(Signal):
    Teager_power = np.zeros(len(Signal))
    # 离散Teager能量算子的公式 = S(n) * S(n) - S(n+1) * S(n-1)
    for i in range(1, len(Signal) - 1):
        Teager_power[i] = Signal[i] * Signal[i] - Signal[i + 1] * Signal[i - 1]
    return Teager_power



#計算數據的均值（Mean）。
def calculate_mean(data):
    """
    計算數據的均值（Mean）。
    :param data: list or np.array, 數據列表或數組
    :return: float, 均值
    """
    return np.mean(data)

#計算數據的變異數（Variance）。
def calculate_variance(data, sample=True):
    """
    計算數據的變異數（Variance）。
    :param data: list or np.array, 數據列表或數組
    :param sample: bool, 是否計算樣本變異數（默認為 True）
    :return: float, 變異數
    """
    ddof = 1 if sample else 0  # ddof=1 表示樣本變異數，ddof=0 表示母體變異數
    return np.var(data, ddof=ddof)

#計算數據的四分位距（Interquartile Range, IQR）。
def calculate_iqr(data):
    """
    計算數據的四分位距（Interquartile Range, IQR）。
    :param data: list or np.array, 數據列表或數組
    :return: float, 四分位距
    """
    q1 = np.percentile(data, 25)  # 第一四分位數 (Q1)
    q3 = np.percentile(data, 75)  # 第三四分位數 (Q3)
    return q3 - q1

#計算數據的偏度（Skewness）。
def calculate_skewness(data):
    """
    計算數據的偏度（Skewness）。
    :param data: list or np.array, 數據列表或數組
    :return: float, 偏度
    """
    return stats.skew(data)

#主要執行的函數
def execute_action(tmplistTemp):
    #global tmplistTemp  # 聲明 tmplistTemp 是全域變數
    if tmplistTemp.size == 0:
        messagebox.showwarning("警告", "請先選取一個檔案")
        return

    #如果未選取檔案，則跳出訊息提示
    if file_label.cget("text") == "Selected File: None":
        messagebox.showinfo("執行結果", "請先選取檔案")
        return
    
    try:
        # 取得使用者輸入的值
        time_window = time_window_entry.get()
        time_window_index = time_window_index_entry.get()
        
        # amplitude = amplitude_slider.get()
        #執行preProcessing(前處理邏輯)
        #前處理結果,msg = pp.preProcessing(int(time_window),int(time_window_index),file_label.cget("text"),int(區間_from_entry.get()),int(區間_to_entry.get()))
        #2025.02.08改為從檔案選取區取得穩定區間的數據
        前處理結果,msg = pp.preProcessing(int(time_window),int(time_window_index),tmplistTemp)
        
        #若前處理結果msg不為空值，則跳出訊息提示
        if msg != "":
            messagebox.showinfo("執行結果", "前處理失敗" + msg)
            return
        特徵值_list = pp.特徵值計算(前處理結果)

        

        # 執行按鈕動作時的提示框
        result = f"Time Window: {time_window}\n"
        result += f"Time Window Index: {time_window_index}\n"
        # result += f"Amplitude: {amplitude}\n"
        result += file_label.cget("text")
        
        #如果有勾選check_box，則執行writeExecl
        if check_var.get() == 1:
            #取得目前檔案的路徑
            path = file_label.cget("text")
            
            #呼叫writeExecl，並將data寫入到test1111.xlsx
            #data是一個list，裡面放的是特徵數據，格式像execl，每一欄是一個特徵
            pp.writeExecl(特徵值_list,"特徵值.xlsx") 
            #alert寫檔成功
            messagebox.showinfo("執行結果", "檔案處理完成，位置在"+path)
        
    except ValueError as e:
        messagebox.showinfo("執行失敗:", e)
        return

    # messagebox.showinfo("執行結果", result)

# 建立主視窗
root = tk.Tk()
root.title("RPPG前處理工具_1.0")
root.geometry("500x500")

# 檔案選取區
browse_button = tk.Button(root, text="選取檔案", command=browse_file)
browse_button.pack(pady=5)

file_label = tk.Label(root, text="Selected File: None", wraplength=300, anchor="w")
file_label.pack(pady=5)

# Time Window 輸入框
time_window_label = tk.Label(root, text="Time Window:")
time_window_label.pack(pady=(10, 0))
time_window_entry = tk.Entry(root)
time_window_entry.pack(pady=5)
# Time Window 輸入框的預設值
time_window_entry.insert(0, "300")

# 第幾個 Time Window 輸入框
time_window_index_label = tk.Label(root, text="第幾個 Time Window:")
time_window_index_label.pack(pady=(10, 0))
time_window_index_entry = tk.Entry(root)
time_window_index_entry.pack(pady=5)
# 第幾個 Time Window 輸入框的預設值
time_window_index_entry.insert(0, "3")

# 穩定區間_起始 = tk.Label(root, text="穩定區間_起始:")
# 穩定區間_起始.pack(pady=(10, 0))
# 穩定區間_起始_entry = tk.Entry(root)
# 穩定區間_起始_entry.pack(pady=5)
# 穩定區間_起始_entry.insert(0, "0")

# 穩定區間_結束 = tk.Label(root, text="穩定區間_結束:")
# 穩定區間_結束.pack(pady=(10, 0))
# 穩定區間_結束_entry = tk.Entry(root)
# 穩定區間_結束_entry.pack(pady=5)
# 穩定區間_結束_entry.insert(0, "999")

區間_from = tk.Label(root, text="區間_from:")
區間_from.pack(pady=(10, 0))
區間_from_entry = tk.Entry(root)
區間_from_entry.pack(pady=5)
區間_from_entry.insert(0, "60")

區間_to = tk.Label(root, text="區間_to:")
區間_to.pack(pady=(10, 0))
區間_to_entry = tk.Entry(root)
區間_to_entry.pack(pady=5)
區間_to_entry.insert(0, "130")

# # 振幅區間滑桿
# amplitude_label = tk.Label(root, text="振幅區間 (0-300):")
# amplitude_label.pack(pady=(10, 0))
# amplitude_slider = tk.Scale(root, from_=0, to=300, orient="horizontal")
# amplitude_slider.pack(pady=5)

#check box - 產生execl檔
check_var = tk.IntVar()
#check box的標籤為"產生execl檔"
check_var.lable = tk.Label(root, text="產生execl檔")
#check box的位置
check_var.lable.pack()
#check box預設為勾選
check_var.set(1)
check_box = tk.Checkbutton(root, variable=check_var)
check_box.pack()

# 執行按鈕
execute_button = tk.Button(root, text="執行", command=lambda: execute_action(browse_file()))
execute_button.pack(pady=20)




# 啟動主迴圈
root.mainloop()