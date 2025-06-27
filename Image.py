from Variables import Variables
import cv2 as cv

class Image:
    def __init__(self, image_path):
        img = cv.imread(image_path)
        self.image = cv.resize(img, Variables.IMAGE_SIZE)
        self.original_image = self.image.copy()
        self.current_zoom = (0, 0, self.image.shape[1], self.image.shape[0])
        self.zoom_factor = Variables.ORIGINAL_ZOOM_FACTOR
        self.brightness = Variables.BRIGHTNESS_STEP
        self.brightness_level = Variables.DEFAULT_BRIGHTNESS_LEVEL
        self.zoom_position = Variables.DEFAULT_ZOOM_POSITION

    def zoom_in(self):
        self.zoom_factor = Variables.ORIGINAL_ZOOM_FACTOR
        zooming_threshold = self.zoom_factor*Variables.ZOOM_THRESHOLD

        x0,y0,w0,h0 = self.current_zoom
        while self.zoom_factor < zooming_threshold:
            self.zoom_factor += Variables.ZOOM_STEP
            h, w = self.image.shape[:2]

            new_w = int(w0 / self.zoom_factor)
            new_h = int(h0 / self.zoom_factor)

            if self.zoom_position == 'center':
                x1 = x0 + (w0 - new_w) // 2
                y1 = y0 + (h0 - new_h) // 2
            elif self.zoom_position == 'top_left':
                x1 = x0
                y1 = y0
            elif self.zoom_position == 'top_right':
                x1 = x0 + w0 - new_w
                y1 = y0
            elif self.zoom_position == 'bottom_left':
                x1 = x0
                y1 = y0 + h0 - new_h
            elif self.zoom_position == 'bottom_right':
                x1 = x0 + w0 - new_w
                y1 = y0 + h0 - new_h

            x2 = x1 + new_w
            y2 = y1 + new_h
            cropped = self.image[y1:y2, x1:x2]
            zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
            cv.imshow("Display window", zoomed_image)
            cv.waitKey(30)
        self.current_zoom = (x1, y1, x2 - x1, y2 - y1)


    def zoom_out(self):
        x0,y0,w0,h0 = self.current_zoom

        h, w = self.image.shape[:2]
        
        self.zoom_factor = w / w0
        center_x = x0 + w0 // 2
        center_y = y0 + h0 // 2

        if self.zoom_factor == 1:
            return

        while self.zoom_factor > Variables.ORIGINAL_ZOOM_FACTOR:
            self.zoom_factor -= Variables.ZOOM_STEP
            if self.zoom_factor < Variables.ORIGINAL_ZOOM_FACTOR:
                self.zoom_factor = Variables.ORIGINAL_ZOOM_FACTOR
            new_w = int(w / self.zoom_factor)
            new_h = int(h / self.zoom_factor)

            x1 = center_x - new_w // 2
            y1 = center_y - new_h // 2

            x1 = max(0, min(w - new_w, x1))
            y1 = max(0, min(h - new_h, y1))
            x2 = x1 + new_w
            y2 = y1 + new_h

            cropped = self.image[y1:y2, x1:x2]
            zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
            cv.imshow("Display window", zoomed_image)
            cv.waitKey(30)
        self.current_zoom = (x1, y1, x2 - x1, y2 - y1)

    def lighting_modification(self):
        temp_img = self.original_image.copy()
        
        self.brightness_level += self.brightness

        self.image = cv.convertScaleAbs(temp_img, beta=self.brightness_level, alpha=1.0)

        cv.imshow("Display window", self.image)