import matplotlib.pyplot as plt

def generate_images(circle_pos, rect_pos, triangle_pos, folder_path, start_idx):
    assert circle_pos != rect_pos and circle_pos != triangle_pos and rect_pos != triangle_pos
    assert circle_pos >= 0 and circle_pos < 3
    assert rect_pos >= 0 and rect_pos < 3
    assert triangle_pos >= 0 and triangle_pos < 3
    
    counter = start_idx
    circle_x_s = range(110, 291, 20)
    circle_r = 40

    if circle_pos == 0:
        circle_y = 100
    if circle_pos == 1:
        circle_y = 200
    if circle_pos == 2:
        circle_y = 300

    rect_x_s = range(70,251, 20)
    rect_w_h = 80

    if rect_pos == 0:
        rect_y = 60
    if rect_pos == 1:
        rect_y = 160
    if rect_pos == 2:
        rect_y = 260

    traingle_x1_s = range(70,251, 20)
    if triangle_pos == 0:
        triangle_y1 = 60
    if triangle_pos == 1:
        triangle_y1 = 160
    if triangle_pos == 2:
        triangle_y1 = 260


    my_dpi = 100
   
    plt.style.use('dark_background')

    for rect_x in rect_x_s:
        for circle_x in circle_x_s:
            for triangle_x1 in traingle_x1_s:
                plt.figure(figsize=(5.2, 5.2), dpi=my_dpi)

                circle = plt.Circle((circle_x, circle_y), radius=circle_r, fc='white')
                plt.gca().add_patch(circle)

                rectangle = plt.Rectangle((rect_x, rect_y), rect_w_h, rect_w_h, fc='white')
                plt.gca().add_patch(rectangle)

                points = [[triangle_x1, triangle_y1], [triangle_x1 + 40, triangle_y1 + 80], [triangle_x1 +80, triangle_y1]]
                polygon = plt.Polygon(points, fc='white')
                plt.gca().add_patch(polygon)

                plt.axis('scaled')
                plt.axis('off')
                plt.xlim([0, 400])
                plt.ylim([0, 400])
                plt.plot()
                plt.savefig(f'{folder_path}image_{counter}', dpi=my_dpi, bbox_inches='tight', pad_inches=0)
                #plt.show()
                plt.close()
                counter = counter + 1

folder = "src/auto_toyimages/"
generate_images(circle_pos=0, rect_pos=1, triangle_pos=2, folder_path=folder, start_idx=0)
generate_images(circle_pos=0, rect_pos=2, triangle_pos=1, folder_path=folder, start_idx=1000)

generate_images(circle_pos=1, rect_pos=0, triangle_pos=2, folder_path=folder, start_idx=2000)
generate_images(circle_pos=1, rect_pos=2, triangle_pos=0, folder_path=folder, start_idx=3000)


generate_images(circle_pos=2, rect_pos=0, triangle_pos=1, folder_path=folder, start_idx=4000)
generate_images(circle_pos=2, rect_pos=1, triangle_pos=0, folder_path=folder, start_idx=5000)