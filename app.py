from io import BytesIO
from threading import Thread
from time import sleep

import numpy as np
import scipy.signal as sig
from PIL import Image
from flask import Flask, Response, render_template, request
from jinja2 import StrictUndefined

WIDTH = 600
HEIGHT = 450
SERVER_UPDATE_S = 0.1
UPDATE_S = 1

# cells = np.zeros((HEIGHT, WIDTH), dtype=bool)
img = Image.open("ic.jpg").resize((WIDTH, HEIGHT))
arr = np.asarray(img)
reds = arr[:, :, 0] >= 128
greens = arr[:, :, 1] >= 128
blues = arr[:, :, 2] >= 128


def update():
    # # index is number of neighbors alive
    rule_alive = np.zeros(8 + 1, np.uint8)  # default all to dead
    rule_alive[[2, 3]] = 1  # alive stays alive <=> 2 or 3 neighbors
    rule_dead = np.zeros(8 + 1, np.uint8)  # default all to dead
    rule_dead[3] = 1  # dead switches to living <=> 3 neighbors

    for cells in [reds, greens, blues]:
        neighbors = sig.convolve2d(
            cells, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            mode='same', boundary='wrap'
        )
        cells[:] = np.where(cells, rule_alive[neighbors], rule_dead[neighbors])


def thread():
    while True:
        sleep(SERVER_UPDATE_S)
        update()


th = Thread(target=thread)
th.start()

app = Flask(__name__)
app.jinja_env.undefined = StrictUndefined
app.jinja_env.globals['UPDATE_S'] = UPDATE_S


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/img')
def cells_png_view():
    stacked = np.where(np.dstack([reds, greens, blues]), 255, 0).astype(np.uint8)
    stream = BytesIO()
    img = Image.fromarray(stacked, mode='RGB')
    img.save(stream, format='png')
    stream.seek(0)
    return Response(stream, mimetype='image/png')


@app.route('/flip', methods=['POST'])
def flip_cell_controller():
    x = request.json.get('x')
    y = request.json.get('y')
    # note reversed order
    for cells in [reds, greens, blues]:
        cells[y, x] = not cells[y, x]
    return request.data


@app.route('/glider', methods=['POST'])
def glider_create_controller():
    x = request.json.get('x')
    y = request.json.get('y')
    for cells in [reds, greens, blues]:
        cells[y-1:y+4, x-1:x+4] = False
        cells[y, x] = True
        cells[y, x+2] = True
        cells[y+1, x+1] = True
        cells[y+1, x+2] = True
        cells[y+2, x+1] = True

        cells[y-1:y+4, x-1:x+4] = np.array([
            [False, False, False, False, False],
            [False, True, False, True, False],
            [False, False, True, True, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ])

    return request.data


if __name__ == '__main__':
    app.run()
