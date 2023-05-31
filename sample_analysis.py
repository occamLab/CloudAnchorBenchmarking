from log_parser import LogParser

parser = LogParser()
parser.analyze_file(parser.all_logs[0], False)
print(parser.get_anchor_poses())
