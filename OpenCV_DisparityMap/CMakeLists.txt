# CMakeList.txt: проект CMake для OpenCV_DisparityMap; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Добавьте источник в исполняемый файл этого проекта.
add_executable (OpenCV_DisparityMap "OpenCV_DisparityMap.cpp" "OpenCV_DisparityMap.h")

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
find_package(OpenCV REQUIRED)
target_link_libraries(OpenCV_DisparityMap ${OpenCV_LIBS})