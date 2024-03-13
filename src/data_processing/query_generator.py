def get_feature_query_for_student(student_id):
    """
    @param student_id: The student Id for whom you want to extract the features from the database.
    @return: Return the query string that can be given to the query processor to be executed.
    """
    # Maintaining a feature list.
    base_feature_map = {

        "activity_details": "SELECT activity_time ,student_id ,activity_inference FROM activity_details ",

        "dinning_details": "SELECT dinning_time ,student_id ,venue_id ,meal_type_id FROM dinning_details  ",

        "call_log_details": "select timestamp as call_time, student_id, 1 as call_recorded  from call_log_details",

        "sms_details": "select timestamp, student_id, 1 as sms_instance from sms_details ",

        "audio_details": "select audio_activity_time, student_id, audio_activity_inference from audio_details ",

        "conversation_details": "select conv_start_timestamp, student_id, conv_duration_min from conversation_details ",

        "dark_details": "select dark_start_timestamp, student_id, dark_duration_min from dark_details ",

        "phonecharge_details": "select start_timestamp, student_id, phonecharge_duration_min from phonecharge_details ",

        "phonelock_details": "select start_timestamp, student_id, phonelock_duration_min from phonelock_details ",

        "gps_details": "select wifi_timestamp as time, student_id, latitude, longitude from gps_details ",

        "sleep_details": "select response_timestamp, student_id, hours_slept, sleep_rating from sleep_details",

    }

    for key in base_feature_map.keys():
        base_feature_map[key] = base_feature_map[key] + " where student_id = " + str(student_id)

    return base_feature_map


def get_stress_query_for_student(student_id):
    """

    @param student_id: StudentId for the stress labels are required.
    @return: Query for StressDetails for the given student.
    """
    stress_query = "select student_id, response_time, adjusted_stress_level as stress_level from stress_details where student_id = "

    return stress_query+str(student_id)
