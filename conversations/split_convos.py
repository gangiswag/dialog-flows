def split_file(input_file, output_file_1, output_file_2, split_ratio=0.8):
    lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    split_index = int(len(lines) * split_ratio)

    file1_lines = lines[:split_index]
    file2_lines = lines[split_index:]

    with open(output_file_1, 'w') as f1:
        f1.writelines(file1_lines)

    with open(output_file_2, 'w') as f2:
        f2.writelines(file2_lines)

if __name__ == "__main__":
    metawoz_test_domains = ["alarm_set", "apartment_finder", "appointment_reminder","bank_bot", "bus_schedule_bot","catalogue_bot", "city_info", "edit_playlist", "event_reserve","guiness_check", "insurance", "library_request", "look_up_info", "music_suggester", "name_suggester", "pet_advice", "scam_lookup", "shopping", "ski_bot", "sports_info", "store_details", "update_calendar", "update_contact", "wedding_planner"]
    metawoz_dev_domains = ["phone_plan", "order_pizza", "movie_listings", "restaurant_picker", "weather_check"]
    multiwoz_domains = ["attraction", "hotel", "restaurant", "taxi", "train"]
    for domain in multiwoz_domains:
        input_file = f'Multiwoz/multiwoz_{domain}.txt'
        output_file_1 = f'Multiwoz/multiwoz_train_{domain}.txt'  # First 80% lines
        output_file_2 = f'Multiwoz/multiwoz_test_{domain}.txt'  # Remaining lines

        split_file(input_file, output_file_1, output_file_2)
