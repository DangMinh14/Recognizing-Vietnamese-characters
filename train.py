import sys
import numpy as np
import cv2
import os
import shutil
def charTraining1(image_path):
    im = cv2.imread(image_path)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    kernel = np.ones((6, 4), np.uint8)
    eroded_img = cv2.erode(gray, kernel, iterations=1)
    _, thresh2 = cv2.threshold(eroded_img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    samples = np.empty((0, 10*10), np.float32)
    responses = []
    keys = [ord('i'), ord('o'), ord('p'), ord('a'), ord('x'), ord('z'), ord('d'), ord('f'), ord('v'), ord('u')]

    # Tạo thư mục để lưu ảnh chữ cái
    output_folder = 'output_characters'
    os.makedirs(output_folder, exist_ok=True)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            [x, y, w, h] = cv2.boundingRect(cnt)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            # Lưu ảnh chữ cái vào thư mục
            character_image_path = os.path.join(output_folder, f'train1_character_{len(responses)}.png')
            cv2.imwrite(character_image_path, roismall)

            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(ord(chr(key)))
                sample = roismall.reshape((1, 10*10))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    print("training complete")

    samples = np.float32(samples)
    responses = np.float32(responses)

    cv2.imwrite("data/train_result1.png", im)
    np.savetxt('data/keys.data', np.array(keys))
    save_data_append(samples, responses)
    cv2.destroyAllWindows()

def charTraining2(image_path):
    im = cv2.imread(image_path)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    kernel = np.ones((6, 4), np.uint8)
    eroded_img = cv2.erode(gray, kernel, iterations=1)
    cv2.waitKey(0)
    thresh2 = cv2.adaptiveThreshold(eroded_img, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    samples = np.empty((0, 10*10), np.float32)
    responses = []
    keys = [ord('i'), ord('o'), ord('p'), ord('a'), ord('x'), ord('z'), ord('d'), ord('f'), ord('v'), ord('u')]

    # Tạo thư mục để lưu ảnh chữ cái
    output_folder = 'output_characters'
    os.makedirs(output_folder, exist_ok=True)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            [x, y, w, h] = cv2.boundingRect(cnt)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            # Lưu ảnh chữ cái vào thư mục
            character_image_path = os.path.join(output_folder, f'train2_character_{len(responses)}.png')
            cv2.imwrite(character_image_path, roismall)

            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(ord(chr(key)))
                sample = roismall.reshape((1, 10*10))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    print("training complete")

    samples = np.float32(samples)
    responses = np.float32(responses)

    cv2.imwrite("data/train_result2.png", im)
    save_data_append(samples, responses)
    np.savetxt('data/keys.data', np.array(keys))
    cv2.destroyAllWindows()
    pass

def save_data_append(samples, responses):
    try:
        # Kiểm tra xem tệp tin đã tồn tại chưa
        if os.path.isfile('data/generalsamples.data') and os.path.isfile('data/generalresponses.data'):
            # Nếu tồn tại, mở tệp tin và append dữ liệu mới
            with open('data/generalsamples.data', 'ab') as f:
                np.savetxt(f, samples)

            with open('data/generalresponses.data', 'ab') as f:
                np.savetxt(f, responses)
        else:
            # Nếu chưa tồn tại, sử dụng np.savetxt để tạo mới tệp tin
            np.savetxt('data/generalsamples.data', samples)
            np.savetxt('data/generalresponses.data', responses)

        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

import shutil

def clear_data():
    try:
        # Xoá tất cả nội dung trong thư mục 'output_characters'
        shutil.rmtree('output_characters')
        # Xoá dữ liệu trong các tệp tin
        with open('data/generalsamples.data', 'w') as file:
            file.write('')
        with open('data/generalresponses.data', 'w') as file:
            file.write('')
        with open('data/keys.data', 'w') as file:
            file.write('')
        
        print("Cleared all data.")
    except Exception as e:
        print(f"Error clearing data: {str(e)}")


def main():
    while True:
        print("------------------------------------")
        print("Which choice?")
        print("1. Train from 'data/train1.png'")
        print("2. Train from 'data/train2.png'")
        print("3. Delete all data")
        print("4. Quit")
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            charTraining1('data/train1.png')
        elif choice == '2':
            charTraining2('data/train2.png')
        elif choice == '3':
            clear_data()
        elif choice == '4':
            print("Exiting...")
            break  # Kết thúc vòng lặp khi bấm số 4
        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()