image_path = 'Thailand.jpg'
image_array = load_image(image_path)
clean_image = median(image_array, ball(3))
edgeMAG = edge_detection(clean_image)
to_print= Image.fromarray((edgeMAG * 255).astype(np.uint8))
to_print.save('my_edges.png')


