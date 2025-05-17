import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

def process_files_recursively(root_folder, cutoff_str, mode):
    try:
        cutoff_time = datetime.strptime(cutoff_str, "%H:%M").time()
    except ValueError:
        messagebox.showerror("錯誤", "請輸入正確的時間格式，例如：15:00")
        return

    total_files = 0
    processed_files = 0

    for folder_path, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".txt"):
                total_files += 1
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    filtered = []
                    for line in lines:
                        try:
                            time_str = line.strip().split(',')[0]
                            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()

                            if mode == "before" and time_obj >= cutoff_time:
                                filtered.append(line)
                            elif mode == "after" and time_obj < cutoff_time:
                                filtered.append(line)
                        except:
                            continue  # 格式錯誤略過

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(filtered)

                    processed_files += 1

                except Exception as e:
                    print(f"❌ 無法處理：{file_path}，錯誤：{e}")

    messagebox.showinfo("完成",
        f"已處理 {processed_files} / {total_files} 個 .txt 檔案\n"
        f"{'刪除之前' if mode == 'before' else '刪除之後'} {cutoff_str} 的資料行。")

# GUI 介面設定
def browse_folder():
    folder = filedialog.askdirectory()
    folder_var.set(folder)

def run_process():
    folder_path = folder_var.get()
    cutoff_str = time_entry.get()
    if not folder_path or not cutoff_str:
        messagebox.showerror("錯誤", "請選擇資料夾並輸入時間。")
        return
    mode = delete_mode.get()
    process_files_recursively(folder_path, cutoff_str, mode)

# Tkinter GUI
root = tk.Tk()
root.title("TXT 時間過濾工具（支援子資料夾 + 前後選擇）")

folder_var = tk.StringVar()
delete_mode = tk.StringVar(value="before")  # 預設刪除之前

# 路徑選擇區
tk.Label(root, text="選擇資料夾：").grid(row=0, column=0, sticky='e')
tk.Entry(root, textvariable=folder_var, width=45).grid(row=0, column=1)
tk.Button(root, text="瀏覽", command=browse_folder).grid(row=0, column=2)

# 時間設定區
tk.Label(root, text="過濾時間（HH:MM）：").grid(row=1, column=0, sticky='e')
time_entry = tk.Entry(root)
time_entry.insert(0, "15:00")
time_entry.grid(row=1, column=1)

# 選擇刪除模式（之前或之後）
tk.Label(root, text="刪除模式：").grid(row=2, column=0, sticky='e')
tk.Radiobutton(root, text="刪除指定時間『之前』的資料", variable=delete_mode, value="before").grid(row=2, column=1, sticky='w')
tk.Radiobutton(root, text="刪除指定時間『之後』的資料", variable=delete_mode, value="after").grid(row=3, column=1, sticky='w')

# 開始按鈕
tk.Button(root, text="開始處理", command=run_process).grid(row=4, column=1, pady=10)

root.mainloop()
