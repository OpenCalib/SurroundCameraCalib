set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

list(APPEND INCLUDES
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            /opt/mpc/3rdparty/protobuf/include
            /opt/mpc/3rdparty/spdlog/include
)
list(APPEND LIBRARIES
            ${OPENGL_LIBRARIES}
)                        
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/perception_config.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/top_view_stitching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_processor_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_processor_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/segment_top_view_stitching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/segment_image_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/segment_image_processor_cuda.cu
)

add_library(image_processor ${SOURCES})
target_include_directories(image_processor PUBLIC ${INCLUDES})
target_link_libraries(image_processor PUBLIC 
    ${LIBRARIES}
    -L/opt/mpc/3rdparty/protobuf/lib/x64_linux protobuf protobuf-lite protoc
)

add_executable(gen_topview_after gen_topview.cpp)

target_link_libraries(gen_topview_after image_processor ${OpenCV_LIBS})