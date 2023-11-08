
import myFunctions
import textwrap
import cv2
import copy
import numpy as np
import os.path


outputimage = []
title = []


#Method Main
def main(imageName):


    #clear some variables
    outputimage.clear()
    title.clear()


    image_colored_BGR = cv2.imread('./images/%s' % imageName, cv2.IMREAD_COLOR)  # Second argument is a flag which specifies the way image should be read

    image_colored_BGR_clone = copy.copy(image_colored_BGR)
    image_colored_RGB = cv2.cvtColor(image_colored_BGR_clone, cv2.COLOR_BGR2RGB)

    image_gray = cv2.cvtColor(image_colored_RGB, cv2.COLOR_BGR2GRAY)

    # print(myFunctions.averagePixels(image_colored_RGB))

    image_gray_downSampled_to_5bits = np.floor_divide(copy.copy(image_gray) , 8)

    testraster = myFunctions.sliding_window_(image_colored_RGB,image_colored_BGR, image_gray_downSampled_to_5bits)

    cv2.namedWindow("RUST DEDECTION")
    cv2.imshow('RUST DEDECTION', testraster)
    cv2.waitKey(0)



if __name__ == "__main__":
    # ========================================================================================================
    # ========================================================================================================
    #                           START OF THE PROGRAM
    # ========================================================================================================
    # ========================================================================================================
    print(textwrap.fill('\n', 80))
    print("*" * 80)
    print(textwrap.fill('SUMMER PROJECT - 2019', 80))
    print(textwrap.fill('IDENTIFICATION OF CORRODED AREAS IN AN IMAGE\n', 80))
    print(textwrap.fill('IMPLEMENTED BY: BASSAM ARNAOUT', 80))
    print(textwrap.fill('SUBMITTED TO: DR. MOHAMMED AYOUB ALAOUI', 80))
    print(textwrap.fill("BISHOP'S UNIVERSITY", 80))
    print(textwrap.fill('\n', 80))
    print(textwrap.fill('The objective of this program the aim is use a data driven approach to evaluate '
                        'corrosion spread across an asset. The input data is a large dataset of high quality '
                        'dense image data. Corroded areas need to be identified in each image.', 80))
    print(textwrap.fill('\n', 80))
    print(textwrap.fill('DATE AUGUST-15-2019', 80))
    print("*" * 80)

    while True:
        print(textwrap.fill('\n\n', 80))
        print(textwrap.fill('Select which image from the below you want to process or '
                            'Type the image name: (Example sample1.jpg):',80))
        print('\n')
        print('1. corrosion1.jpg            2. corrosion2.jpg')
        print('3. corrosion3.jpg            4. corrosion4.jpg')
        print('5. corrosion5.jpg            6. corrosion6.jpg')
        print('7. corrosion7.jpg            8. corrosion8.jpg')
        print('9. corrosion9.jpg            10. white_glacier.jpg')
        print('11. car3.jpg                 12. sample1.jpg')
        print('13. sample2.jpg              14. sample3.jpg')
        print('15. sample4.jpg              16. sample5.jpeg')
        print('17. pipes1.jpeg              18. lion.png')
        print('Q.  QUIT')

        userInput = input("\nEnter your selection or Type image name: ")
        print('\n')

        imageName = myFunctions.nameOfImageFile(userInput)

        if userInput == 'Q' or userInput == 'q':
            break
        else:
            if imageName == 'nothing':
                #check if the entered string from user is a file name and it exist
                if not os.path.isfile('./images/%s' % userInput,):
                    # ignore if no such file is present.

                    print('\nFILE ("%s") COULD NOT BE OPENED...\n' %userInput)
                    print("-" * 80)
                    pass

                else:
                    main(userInput)
                    print('done')
            else:
                main(imageName)
                # print('done')