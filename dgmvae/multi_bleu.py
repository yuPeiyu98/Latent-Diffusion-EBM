import os
import subprocess
import sys

root_dir = os.getcwd()
perl_path = os.path.join(root_dir, "multi-bleu.perl")

def multi_bleu_perl(dir_name):
    print("Runing multi-bleu.perl")
    dir = os.path.join(root_dir, dir_name)
    for file in os.listdir(dir):
        if "-test-greedy.txt.txt" in file or file[-16:] == "-test-greedy.txt":
            file_name = os.path.join(dir, file)
            f = open(file_name, "r")

            hyps_f = open(os.path.join(dir, "hyp"), "w")
            refs_f = open(os.path.join(dir, "ref"), "w")

            if not file:
                print("Open file error!")
                exit()
            for line in f:
                if line[:7] == "Target:":
                    refs_f.write(line[7:].strip() + "\n")
                if line[:8] == "Predict:":
                    hyps_f.write(line[8:].strip() + "\n")
                    # hyps += line[8:]

            hyps_f.close()
            refs_f.close()
            f.close()
            p = subprocess.Popen(["perl", perl_path, os.path.join(dir, "ref")], stdin=open(os.path.join(dir, "hyp"), "r"),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            while p.poll() == None:
                pass
            print("multi-bleu.perl return code: ", p.returncode)
            # os.remove(os.path.join(dir, "hyp"))
            os.remove(os.path.join(dir, "ref"))

            fout = open(file_name, "a")
            for line in p.stdout:
                line = line.decode("utf-8")
                if line[:4] == "BLEU":
                    sys.stdout.write(line)
                    fout.write(line)


            # return p.returncode


if __name__ == "__main__":
    dir_name = sys.argv[1]
    print(dir_name)
    multi_bleu_perl(dir_name)
