import matplotlib.pyplot as plt

rect_y = 160
rect_x_s = range(70,251, 20)
rect_w_h = 80

circle_y = 300
circle_x_s = range(110, 291, 20)
circle_r = 40

triangle_y1_y3 = 60
triangle_y2 = 140
traingle_x1_s = range(70,251, 20)
traingle_x2_s = range(110, 291, 20)

my_dpi = 100
counter = 0

plt.style.use('dark_background')

for rect_x in rect_x_s:
    for circle_x in circle_x_s:
        for (triangle_x1, triangle_x2) in zip(traingle_x1_s, traingle_x2_s):
            # print("Rectangle:", rect_x, rect_y)
            
            # print("Circle:", circle_y, circle_x)

            # print("Triangle", triangle_x1, triangle_x2 )
            plt.figure(figsize=(5.2, 5.2), dpi=my_dpi)
            #plt.axes()
            circle = plt.Circle((circle_x, circle_y), radius=circle_r, fc='white')
            plt.gca().add_patch(circle)

            rectangle = plt.Rectangle((rect_x, rect_y), rect_w_h, rect_w_h, fc='white')
            plt.gca().add_patch(rectangle)

            points = [[triangle_x1, triangle_y1_y3], [triangle_x2, triangle_y2], [triangle_x1 +80, triangle_y1_y3]]
            polygon = plt.Polygon(points, fc='white')
            plt.gca().add_patch(polygon)

            plt.axis('scaled')
            plt.axis('off')
            plt.xlim([0, 400])
            plt.ylim([0, 400])
            plt.plot()
            plt.savefig(f'src/auto_toyimages/image_{counter}', dpi=my_dpi, bbox_inches='tight', pad_inches=0)
            #plt.show()
            plt.close()
            counter = counter + 1
