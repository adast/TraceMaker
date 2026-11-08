find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
# One `route` run as a library function: shared by the CLI and the Python bindings (bindings/); and the open-connection causes (doc 05 §55).
add_library(tm_app_route STATIC route_job.cpp causes.cpp)
add_library(tm::app_route ALIAS tm_app_route)
target_link_libraries(tm_app_route PUBLIC tm::route tm::analysis nlohmann_json::nlohmann_json PRIVATE tm::advice tm::report tm::passes tm::crules tm::core tm::gpu tm::io tm::learn tm::server tm::replay tm::viz)
tm_set_warnings(tm_app_route)

# The options screen (`tracemaker tui`): the model is free of terminal and CLI code, so the unit tests link it.
add_library(tm_app_tui STATIC tui.cpp tui_term.cpp)
add_library(tm::app_tui ALIAS tm_app_tui)
target_include_directories(tm_app_tui PUBLIC ${PROJECT_SOURCE_DIR}/src)
tm_set_warnings(tm_app_tui)

add_executable(tracemaker main.cpp inspect.cpp defects.cpp)
target_link_libraries(tracemaker PRIVATE tm::analysis tm::advice tm::app_route tm::app_tui tm::report tm::passes tm::crules tm::core tm::gpu tm::io tm::drc tm::route tm::learn tm::server tm::session tm::viz CLI11::CLI11 nlohmann_json::nlohmann_json)
tm_set_warnings(tracemaker)

add_library(tm_board_batch STATIC board_batch.cpp)
target_link_libraries(tm_board_batch PUBLIC tm::app_route PRIVATE tm::gpu)
tm_set_warnings(tm_board_batch)
add_executable(tracemaker-batch batch_main.cpp)
target_link_libraries(tracemaker-batch PRIVATE tm_board_batch CLI11::CLI11)
tm_set_warnings(tracemaker-batch)
