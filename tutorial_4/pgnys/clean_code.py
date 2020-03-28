import argparse
import cv2
import os.path


# Example of a mixing that takes image and returns processed image and
# information
# if program should continue execution
def close(img, out_file):
    running = False
    return img, running


def scale_up(img, out_file):
    running = True
    rows, cols, chans = map(int, img.shape)
    img = cv2.pyrUp(img, dstsize=(2 * cols, 2 * rows))
    return img, running


def scale_down(img, out_file):
    running = True
    rows, cols, chans = map(int, img.shape)
    img = cv2.pyrDown(img, dstsize=(cols // 2, rows // 2))
    return img, running


def save(img, out_file):
    running = True
    file_count = 0
    while True:
        file_name = '{}_{}.png'.format(out_file, file_count)
        if os.path.isfile(file_name):
            file_count += 1
            continue
        cv2.imwrite(file_name, img)
        print('Saved image to {}'.format(file_name))
        break

    return img, running


def main(input_file, out_file):
    img = cv2.imread(input_file)
    if img is None:
        print('Unable to open image at {}'.format(input_file))
        return

    keymap = {ord('q'): close,
              ord('o'): scale_down,
              ord('p'): scale_up,
              ord('s'): save
              }

    running = True
    while running:
        cv2.imshow('Our image', img)
        k = cv2.waitKey(0)
        if k in keymap.keys():
            img, running = keymap[k](img, out_file)
        else:
            print('Unasigned key {}'.format(chr(k)))


def parse_arguments():
    desc = 'Image scaling software for MWR classes.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help='Path to input image')
    parser.add_argument('-o', '--out', help='First part of output file name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.input, args.out)
