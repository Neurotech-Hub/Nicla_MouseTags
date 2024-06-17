# Edge Impulse - OpenMV Object Detection Example

import sensor, image, time, os, tf, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.85

try:
    # Load built in model
    labels, net = tf.load_builtin_model('trained')
except Exception as e:
    raise Exception(e)

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255, 255, 255), #0
    (0,   255,   0),  # green
    (  255, 0,   0),  # red
    (255, 255,   255),  # white
    (  0,   0, 255),  # Blue
    (255,   0, 255),  # Magenta
    (  0, 255, 255),  # Cyan
    (255, 255, 255),  # White
]

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # Dictionary to keep track of the highest confidence detection for each class
    highest_confidence_detections = {}

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects
    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?

        for d in detection_list:
            confidence = d.output()
            if i not in highest_confidence_detections or confidence > highest_confidence_detections[i][0]:
                highest_confidence_detections[i] = (confidence, d.rect())

    for i, (confidence, rect) in highest_confidence_detections.items():
        [x, y, w, h] = rect
        center_x = math.floor(x + (w / 2))
        center_y = math.floor(y + (h / 2))
        color = colors[i % len(colors)]

        print("********** %s **********" % labels[i])
        print('x: %d, y: %d, confidence: %.2f' % (center_x, center_y, confidence))

        # Draw the circle with the corresponding color
        img.draw_circle(center_x, center_y, 12, color=color, thickness=3)

        # Draw the confidence score on the image
        img.draw_string(x, y-20, "%.2f" % (confidence), color=color, scale=1.5)

    print(clock.fps(), "fps", end="\n\n")
