
from optparse import OptionParser, OptionValueError
import cv2
import sys

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Video FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()


# Check that a file is passed as an argument.
if options.filename is None:
    print "Pass a file as an argument. For example -f car.avi"
    parser.print_help()
    sys.exit(1)

backsub = cv2.BackgroundSubtractorMOG()
capture = cv2.VideoCapture(options.filename)

keep_processing = True
# Set horizon line (as a fraction of the height).
# 0.3 means 30% from the top
HORIZON_LINE_RATIO = 0.6

while(keep_processing):
    ret, frame = capture.read()

    # When the stream has no more frames, stop processing.
    if frame is None:
        keep_processing = False
        break

    height, width, __ = frame.shape
    horizon = int(height * HORIZON_LINE_RATIO)
    fgmask = backsub.apply(frame, None, 0.01)
    contours, hierarchy_vector = cv2.findContours(fgmask.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)

    hierarchy = []

    if hierarchy_vector is not None:
        hierarchy = hierarchy_vector[0]
 
    # Draw counting line.
    cv2.line(frame, (0, horizon), (width, horizon), (255, 255, 255), 1)

    # Iterate over the contours to draw the rectangles.
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)

        if y < horizon < y + h:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cv2.putText(frame, "x=%s" % x, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (255, 0, 0), 2)

    cv2.imshow("Track", frame)
    cv2.imshow("background sub", fgmask)

    # Way for user input, if the user presses q then stop processing.
    key = cv2.waitKey(100)
    if key == ord('q'):
        keep_processing = False
        break
