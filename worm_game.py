import curses
from random import randint

# Directions
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    sh, sw = stdscr.getmaxyx()
    box = [[3, 3], [sh - 3, sw - 3]]

    for i in range(box[0][1], box[1][1]):
        stdscr.addstr(box[0][0], i, '#')
        stdscr.addstr(box[1][0], i, '#')
    for i in range(box[0][0], box[1][0]):
        stdscr.addstr(i, box[0][1], '#')
        stdscr.addstr(i, box[1][1], '#')

    worm = [[sh // 2, sw // 2 + i] for i in range(3)][::-1]
    direction = LEFT

    food = [randint(box[0][0] + 1, box[1][0] - 1),
            randint(box[0][1] + 1, box[1][1] - 1)]
    stdscr.addstr(food[0], food[1], '*')

    while True:
        stdscr.addstr(worm[0][0], worm[0][1], 'O')
        for y, x in worm[1:]:
            stdscr.addstr(y, x, 'o')

        key = stdscr.getch()
        if key == curses.KEY_UP and direction != DOWN:
            direction = UP
        elif key == curses.KEY_DOWN and direction != UP:
            direction = DOWN
        elif key == curses.KEY_LEFT and direction != RIGHT:
            direction = LEFT
        elif key == curses.KEY_RIGHT and direction != LEFT:
            direction = RIGHT
        elif key == ord('q'):
            break

        head = [worm[0][0] + direction[0], worm[0][1] + direction[1]]

        if (head in worm or
            head[0] in (box[0][0], box[1][0]) or
            head[1] in (box[0][1], box[1][1])):
            msg = 'Game Over!'
            stdscr.nodelay(False)
            stdscr.addstr(sh // 2, sw // 2 - len(msg) // 2, msg)
            stdscr.getch()
            break

        worm.insert(0, head)

        if head == food:
            food = None
            while food is None:
                nf = [randint(box[0][0] + 1, box[1][0] - 1),
                      randint(box[0][1] + 1, box[1][1] - 1)]
                if nf not in worm:
                    food = nf
            stdscr.addstr(food[0], food[1], '*')
        else:
            tail = worm.pop()
            stdscr.addstr(tail[0], tail[1], ' ')

    stdscr.nodelay(False)
    stdscr.timeout(-1)


if __name__ == '__main__':
    curses.wrapper(main)
