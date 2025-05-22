import math

RADIUS = 2
LINE_WIDTH = 4


def draw_arrow(draw, points, radius=RADIUS, line_width=LINE_WIDTH, color='red'):
    
    total_length = 0
    for point1, point2 in zip(points[:-2], points[1:-1]):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=color)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        if not (x1 == x2 and y1 == y2):
            total_length += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    x1, y1 = points[-2,0], points[-2,1]
    x2, y2 = points[-1,0], points[-1,1]
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    
    total_length += math.sqrt((dx)**2 + (dy)**2)
    arrowhead_length = line_width * 3
    if total_length < arrowhead_length:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        return total_length
    
    end_x = x2 - arrowhead_length * 0.8 * math.cos(angle)
    end_y = y2 - arrowhead_length * 0.8 * math.sin(angle)
    
    draw.line([(x1, y1), (end_x, end_y)], fill=color, width=line_width)
    
    left_angle = angle + math.pi / 6
    right_angle = angle - math.pi / 6
    
    left_arrowhead = (x2 - arrowhead_length * math.cos(left_angle),
                      y2 - arrowhead_length * math.sin(left_angle))
    right_arrowhead = (x2 - arrowhead_length * math.cos(right_angle),
                       y2 - arrowhead_length * math.sin(right_angle))
    
    draw.polygon([(x2, y2), left_arrowhead, right_arrowhead], fill=color)
    return total_length


def draw_dotted(draw, points, radius=RADIUS, color='red'):
    
    total_length = 0
    for point1, point2 in zip(points[:-1], points[1:]):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=color)
        if not (x1 == x2 and y1 == y2):
            total_length += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
    x1, y1 = points[-1,0], points[-1,1]
    draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=color)
    
    return total_length

