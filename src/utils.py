import cv2

def draw_text_with_outline(frame, text, position, font, font_scale, text_colour, outline_colour, thickness):
    """
    Draw text with an outline for better visibility.
    """
    x, y = position
    # Draw outline (background)
    cv2.putText(frame, text, (x, y), font, font_scale, outline_colour, thickness + 4, cv2.LINE_AA)
    # Draw text (foreground)
    cv2.putText(frame, text, (x, y), font, font_scale, text_colour, thickness, cv2.LINE_AA)
