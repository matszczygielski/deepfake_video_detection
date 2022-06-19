import csv


class WriterCSV:
    def __init__(self, filename) -> None:
        self.filename = filename

        self.is_header_written = False


    def write(self, content: dict) -> None:
        if not self.is_header_written:
            headers = []
            for key, value in content.items():
                headers.append(key)

            with open(self.filename, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(headers)

            self.is_header_written = True

        features = []
        for key, value in content.items():
            features.append(value)

        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(features)

