import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class PandasModel:
    def __init__(self, df):
        self.df = df

    def get_model(self):
        model = PandasModel(self.df)
        return model

    def data(self, index):
        if index.isValid():
            return str(self.df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if orientation == 1 and role == 0:
            return str(self.df.columns[section])
        return None

    def rowCount(self, parent):
        return len(self.df)

    def columnCount(self, parent):
        return len(self.df.columns)

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def display_info(self):
        model = PandasModel(self.df)

        table = ttk.Treeview(root)
        table['columns'] = tuple(self.df.columns)
        table['show'] = 'headings'

        for column in self.df.columns:
            table.heading(column, text=column)

        for i in range(len(self.df)):
            table.insert("", "end", values=list(self.df.iloc[i]))

        table.pack()

    def plot_data(self):
        plot_choice = input("Do you want plotting? (yes/no): ")

        if plot_choice.lower() == "yes":
            plot_type = input("Plotting type 1 classic, 2 customized: ")

            if plot_type == "1":
                self.plot_classic()
            elif plot_type == "2":
                self.plot_customized()
            else:
                print("Invalid plot type.")
        else:
            print("Analysis finished.")

    def plot_classic(self):
        print("Plotting started:")
        self.df.plot()
        plt.grid()
        plt.show()
        input("Press Enter to continue...")

    def plot_customized(self):
        x = input("X:")
        y = input("Y:")
        kind = input("Kind:")
        print("Plotting started:")
        self.df.plot(kind=str(kind), x=str(x), y=str(y))
        plt.grid()
        plt.show()
        input("Press Enter to continue...")

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values
        return torch.tensor(sample, dtype=torch.float32)

def analyze_data(df):
    try:
        analyzer = DataAnalyzer(df)
        analyzer.display_info()
        analyzer.plot_data()

        # Additional PyTorch operations can be added here
        # Example:
        # torch_data = CustomDataset(df)
        # dataloader = DataLoader(torch_data, batch_size=1, shuffle=True)
        # for batch in dataloader:
        #     print(batch)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    root.title("Advanced Data Analysis App")
    root.geometry("800x600")

    file_path = filedialog.askopenfilename(title="Select File", filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")])

    if not file_path:
        print("No file selected.")
        return

    file_type = file_path.split(".")[-1]

    try:
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError("Invalid file type.")

        analyze_data(df)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    root.mainloop()

if __name__ == "__main__":
    main()
