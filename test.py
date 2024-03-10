import cv2
import numpy as np

def detect_objects(image_path, result_image_path):
    keys = np.loadtxt('data/keys.data', np.int32)
    samples = np.loadtxt('data/generalsamples.data', np.float32)
    responses = np.loadtxt('data/generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    im = cv2.imread(image_path)
    im = cv2.resize(im, (1600, 800))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((8, 2), np.uint8)
    kernel1 = np.ones((1, 2), np.uint8)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    erosed_img = cv2.erode(thresh, kernel1, iterations=1)
    dilated_img = cv2.dilate(erosed_img, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 1:
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 10 * 10))
                roismall = np.float32(roismall)

                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)

                if dists[0] < 1000000:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lưu ảnh với bounding box đã được vẽ
    cv2.imwrite(result_image_path, im)

    cv2.imshow('Result', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    while True:
        print("Choose an option:")
        print("1. Picture 1")
        print("2. Picture 2")
        print("3. Picture 3")
        print("4. Picture 4")
        print("5. Picture 5")
        print("0. Quit")

        choice = input("Enter your choice (0-5): ")

        if choice == '0':
            break
        elif choice in ['1']:
            image_path = 'data/test{}.png'.format(choice)
            result_image_path = 'data/test_result{}.jpg'.format(choice)
            detect_objects(image_path, result_image_path)
        elif choice in ['2', '3', '4', '5']:
            image_path = 'data/test{}.jpg'.format(choice)
            result_image_path = 'data/test_result{}.jpg'.format(choice)
            detect_objects(image_path, result_image_path)
        else:
            print("Invalid choice. Please enter a number between 0 and 5.")

if __name__ == "__main__":
    main()
