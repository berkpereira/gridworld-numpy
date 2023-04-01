import tkinter as tk

class GridGUI(tk.Frame):
    def __init__(self, grid_size, landing_zone):
        super().__init__()
        self.grid_size = grid_size
        self.landing_zone = landing_zone
        self.grid_markers = [[0] * grid_size for _ in range(grid_size)]

        self.create_widgets()

    def create_widgets(self):
        self.grid = tk.Canvas(self, width=400, height=400)
        self.grid.pack()

        self.reset_button = tk.Button(self, text="Reset", command=self.reset)
        self.reset_button.pack()

        # Draw grid lines
        for i in range(self.grid_size):
            x = i * 400 // self.grid_size
            self.grid.create_line(x, 0, x, 400)
            self.grid.create_line(0, x, 400, x)

        # Draw landing zone marker
        lz_x, lz_y = self.landing_zone
        self.grid_markers[lz_x][lz_y] = 1
        x0 = lz_x * 400 // self.grid_size
        y0 = lz_y * 400 // self.grid_size
        x1 = (lz_x + 1) * 400 // self.grid_size
        y1 = (lz_y + 1) * 400 // self.grid_size
        self.grid.create_rectangle(x0, y0, x1, y1, fill="green")
        self.grid.create_text(x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2, text=f'{lz_x}, {lz_y}')

        # Bind click event to grid
        self.grid.bind("<Button-1>", self.mark)

    def mark(self, event):
        x = event.x * self.grid_size // 400
        y = event.y * self.grid_size // 400
        if self.grid_markers[x][y] == 0:
            self.grid_markers[x][y] = 2
            x0 = x * 400 // self.grid_size
            y0 = y * 400 // self.grid_size
            x1 = (x + 1) * 400 // self.grid_size
            y1 = (y + 1) * 400 // self.grid_size
            self.grid.create_rectangle(x0, y0, x1, y1, fill="red")
            self.grid.create_text(x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2, text=f'{x}, {y}')

    def reset(self):
        self.grid.delete("all")
        self.grid_markers = [[0] * self.grid_size for _ in range(self.grid_size)]
        lz_x, lz_y = self.landing_zone
        self.grid_markers[lz_x][lz_y] = 1
        x0 = lz_x * 400 // self.grid_size
        y0 = lz_y * 400 // self.grid_size
        x1 = (lz_x + 1) * 400 // self.grid_size
        y1 = (lz_y + 1) * 400 // self.grid_size
        self.grid.create_rectangle(x0, y0, x1, y1, fill="green")
        self.grid.create_text(x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2, text=f'{lz_x}, {lz_y}')
        # Draw grid lines
        for i in range(self.grid_size):
            x = i * 400 // self.grid_size
            self.grid.create_line(x, 0, x, 400)
            self.grid.create_line(0, x, 400, x)

if __name__ == "__main__":
    # Inputs here.
    grid_size = 13
    landing_zone = [6, 6]
    

    root = tk.Tk()
    root.title("GridGUI")
    app = GridGUI(grid_size, landing_zone)
    app.pack()
    root.mainloop()
