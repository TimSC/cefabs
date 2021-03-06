SET(QT_USE_QTMAIN 1)
INCLUDE(${QT_USE_FILE})

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}  
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/gpu 
    ${PROJECT_SOURCE_DIR}/util )

FILE(GLOB_RECURSE sources *.cpp *.h *.ui *.qrc)
IF(WIN32)
    SET(sources ${sources} cefabs.rc)
ENDIF()
IF(APPLE)
    SET(sources ${sources} cefabs.icns)
    SET_SOURCE_FILES_PROPERTIES( cefabs.icns
        PROPERTIES MACOSX_PACKAGE_LOCATION Resources )
ENDIF()

QT4_AUTO_WRAP( sources ${sources} )
CUDA_ADD_EXECUTABLE( cefabs WIN32 MACOSX_BUNDLE ${sources} )

SOURCE_GROUP(src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$")
SOURCE_GROUP(generated REGULAR_EXPRESSION "cxx$|ui_")

IF(MSVC)
    IF(MSVC_IDE)
        SET_TARGET_PROPERTIES( cefabs
            PROPERTIES
            COMPILE_FLAGS "/FIstable.h /Yustable.h" )
            
        SET_SOURCE_FILES_PROPERTIES( stable.cpp
            PROPERTIES
            COMPILE_FLAGS "/Ycstable.h" )
    ELSE()
        SET_TARGET_PROPERTIES( cefabs
            PROPERTIES
            COMPILE_FLAGS "/FIstable.h" )
    ENDIF()
ELSE()
    
    SET_TARGET_PROPERTIES( cefabs
        PROPERTIES
        COMPILE_FLAGS "-include stable.h" )
ENDIF()

IF(APPLE)
    SET_TARGET_PROPERTIES( cefabs
        PROPERTIES
        MACOSX_BUNDLE_ICON_FILE "cefabs.icns" )
ENDIF()


TARGET_LINK_LIBRARIES( cefabs ${QT_LIBRARIES} ${FFmpeg_LIBRARIES} util gpu )

INSTALL( TARGETS cefabs
    BUNDLE DESTINATION .
    RUNTIME DESTINATION .
    CONFIGURATIONS Debug Release )

FIXUP_BUNDLE(cefabs)
