import json
from argparse import ArgumentParser
from pathlib import Path
from lpinstance import LPInstance
from timer import Timer

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    path = Path(input_file)
    filename = path.name

    timer = Timer()
    timer.start()

    instance = LPInstance(str(input_file))
    instance.solve()

    timer.stop()

    output_dict = {"Instance": filename,
                   "Time": f"{timer.getTime():.2f}",
                   "Result": instance.objective_value,
                   "Solution": "OPT"}

    print(json.dumps(output_dict))
