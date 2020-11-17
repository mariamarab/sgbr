import os
import datetime
import struct
import ast
import numpy as np
from numpy import (array, float64, power, log)
from path import Path
from egpr_utils import *
from optparse import OptionParser
from mpgraph import MPGraphNode, MPGraph

parser = OptionParser()
parser.add_option('--biosample', dest='biosample', metavar='STR')
parser.add_option('--resolution', dest='resolution', help="the specified resolution in bp", metavar='N')
(options, args) = parser.parse_args()

biosample = options.biosample
resolution = options.resolution

class EGPRInstance:
    def __init__(self, biosample, resolution = 1000):
        self.biosample = biosample
        
        # Resolution
        self.resolution = resolution
        #self.virtual_resolution = resolution

        # File names
        self.outdir_path = biosample if biosample != None else self.createOutdir()
        self.out_files = self.outFiles(self.outdir_path)
        self.in_files = self.inFiles()

        # Init training
        if not os.path.isfile(self.out_files['train']['windows_filename']):
            cmd = ["segway", "train-init", 
            "--resolution=" + str(self.resolution),
            "--include-coords=" + self.in_files['train']['include_coords_filename'], 
            self.in_files['train']['genomedata_filename'], 
            self.out_files['train']['traindir_path']]
            runCommand(cmd)

        # Get windows
        self.windows, self.num_frames = coords_file(self.out_files['train']['windows_filename'], self.resolution)

        # MP Graph variables
        self.node_frames = None
        self.neighbours = None

        # Other params
        self.num_egpr_iters = 5
        self.num_labels = 2
        self.num_segs = 2
        self.egpr_nu = 1.0
        self.egpr_mu = 1.0
        self.egpr_alpha = 1.0
        self.measure_prop_num_iters = 100
        self.mp_weight = 1

    def createOutdir(self):
        now = datetime.datetime.now()
        month = now.strftime("%m-%Y")
        date = now.strftime("%d-%H%M%S")
        outdir_path = file_path('../res/' + month + '/traindir' + "_" + date + os.extsep + 'res' + str(self.virtual_resolution))
        return outdir_path
    
    def inFiles(self):
        datadir_path = "../data/res1"
        hic_datadir_path = "../data/res{}".format(self.resolution)
        in_files = {
            'train': {
                # 'genomedata_filename': Path(datadir_path) / '{biosample}.res{resolution}.genomedata'.format(biosample = self.biosample, resolution = self.virtual_resolution),
                'genomedata_filename': Path(datadir_path) / '{biosample}.genomedata'.format(biosample = self.biosample),
                # 'include_coords_filename': Path(datadir_path) / 'encodePilotRegions.hg19.res{resolution}.test.bed'.format(resolution = self.virtual_resolution),
                'include_coords_filename': Path(datadir_path) / 'encodePilotRegions.hg19.bed',
                # 'exclude_coords_filename': Path(datadir_path) / 'ENCFF001TDO.bed'.format(resolution = self.virtual_resolution),
            },
            'mp': {
                # 'hic_filename': Path(hic_datadir_path) / '{biosample}_combined.spline_pass1.res1000.significances.res{resolution}.hic'.format(biosample = self.biosample, resolution = self.virtual_resolution),
                'hic_filename': Path(hic_datadir_path) / '{biosample}_combined.spline_pass1.res1000.significances.hic'.format(biosample = self.biosample),
            },
        }
        # for task in list(in_files):
        #     for input_file in in_files[task]:
        #         if not os.path.exists(input_file):
        #             exit(input_file + " does not exist.")
        return in_files

    def outFiles(self, outdir_path):
        train_outdir_path = file_path(Path(outdir_path) / 'train')
        mp_outdir_path = file_path(Path(outdir_path) / 'mp')
        ve_outdir_path = file_path(Path(outdir_path) / 've')
        post_outdir_path = file_path(Path(outdir_path) / 'post')
        out_files = {
            'train': {
                'traindir_path': train_outdir_path,
                'windows_filename': Path(train_outdir_path) / "window.bed",
                },
            'mp': {
                'mp_outdir_path': mp_outdir_path,
                'mp_graph_filename': Path(mp_outdir_path) / 'mp_graph',
                'mp_trans_filename': Path(mp_outdir_path) / 'mp_trans',
                'mp_label_filename': Path(mp_outdir_path) / '{egpr_iter}.mp_label',
                'mp_post_filename': Path(mp_outdir_path) / 'post.mp_label',
                'mp_obj_filename': Path(mp_outdir_path) / 'mp_obj',
                'mp_graph_neighbours_filename': Path(mp_outdir_path) / 'mp_graph_neighbours',
            },
            've': {
                've_outdir_path': ve_outdir_path,
                'virtual_evidence_filename': Path(ve_outdir_path) / 've.bed',
            },
            'post': {
                'post_outdir_path': post_outdir_path,
                'post_tmpl_path': Path(post_outdir_path) / 'egpr-{egpr_iter}',
                'post_filename': Path(post_outdir_path) / 'egpr-{egpr_iter}' / 'posterior'/ 'posterior{label_index}.{window_index}.bed',
            },
        }
        return out_files

    def write_mp_graph(self):
        resolution = self.resolution
        windows = self.windows
        num_frames = self.num_frames
        threshold = 1.0
        max_inters= -1
        sorted=False
        interaction_type="intra"
        max_dist= -1

        in_files = self.in_files['mp']
        out_files = self.out_files['mp']

        fithic_filename = in_files['hic_filename']
        mp_graph_filename = out_files['mp_graph_filename']
        mp_trans_filename = out_files['mp_trans_filename']
        mp_graph_neighbours_filename = out_files['mp_graph_neighbours_filename']

        neighbours = [[] for i in range(num_frames)]
        num_inters = 0
        num_skipped = 0

        # List of frames which will have nodes in the final graph
        node_frames = []
        with maybe_gzip_open(fithic_filename) as f:
            for line_index, line in enumerate(f):
                if (line_index % 1000000) == 0: print(line_index, num_inters)
                if num_inters == max_inters: break
                if ("contactCount" in line) or ("fragmentMid" in line): # skip header line
                    print("skipping: ", line.strip())
                    continue
                line = line.split()
                chrom1 = line[0]
                pos1 = int(line[1])
                chrom2 = line[2]
                pos2 = int(line[3])
                contact_count = int(line[4])
                if len(line) > 5:
                    pvalue = float(line[5])
                    qvalue = float(line[6])
                else:
                    pvalue, qvalue = 1.0, 1.0

                if (interaction_type == "intra") and (chrom1 != chrom2):
                    num_skipped += 1
                    continue

                if (max_dist != -1) and (abs(pos1-pos2) > max_dist):
                    num_skipped += 1
                    continue

                pvalue = max(pvalue, 1e-100)
                if pvalue >= threshold:
                    if sorted:
                        print("Stopping after finding an interaction with pvalue=%s >= threshold=%s.  If intreractions aren't sorted, do not specify --sorted" % (pvalue, threshold))
                        print("Line was:", line)
                        print
                        break
                    else:
                        num_skipped += 1
                        continue

                window, frame1 = get_frame_index(windows, resolution, chrom1, pos1)
                window, frame2 = get_frame_index(windows, resolution, chrom2, pos2)
                if ((frame1 != -1) and (frame2 != -1)):
                    weight = -log(threshold * pvalue)
                    if frame1 not in node_frames:
                        node_frames.append(frame1)
                    if frame2 not in node_frames:
                        node_frames.append(frame2)
                    neighbours[frame1].append((node_frames.index(frame2), weight))
                    neighbours[frame2].append((node_frames.index(frame1), weight))
                    num_inters += 1
                else:
                    pass

        print("Total number of interactions written:", num_inters)
        print("Total number of interactions skipped:", num_skipped)

        nodes = []
        print(len(node_frames))
        for frame_index in range(num_frames):
            nodes.append(MPGraphNode().init(frame_index, neighbours[frame_index]))

        graph = MPGraph().init(nodes)
        with open(mp_graph_filename, "wb") as f:
            graph.write_to(f)

        self.node_frames = range(len(neighbours))
        self.neighbours = neighbours

        label_fmt = "I"
        with open(mp_trans_filename, "wb") as f:
            for i in range(num_frames):
                f.write(struct.pack(label_fmt, 1)) 

        with open(mp_graph_neighbours_filename, "w") as f:
            f.write(str(neighbours)) 

    def write_segway_posteriors(self, egpr_iter_index):
        out_files = self.out_files['train']
        in_files = self.in_files['train']
        genomedata_filename = in_files['genomedata_filename']
        traindir_path = out_files['traindir_path']
        include_coords_filename = in_files['include_coords_filename']
        #exclude_coords_filename = in_files['exclude_coords_filename']
        out_files = self.out_files['post']
        postdir_path = out_files['post_tmpl_path'].format(egpr_iter = egpr_iter_index)

        cmd = ["segway", "train-run-round", genomedata_filename, traindir_path]
        runCommand(cmd)

        cmd = ["segway", "train-finish", genomedata_filename, traindir_path]
        runCommand(cmd)

        cmd = ["segway",  "posterior-init", \
        "--include-coords=" + include_coords_filename, \
        genomedata_filename, \
        traindir_path, \
        postdir_path]
        runCommand(cmd)

        cmd = ["segway", "posterior-run", genomedata_filename, traindir_path, postdir_path]
        runCommand(cmd)


    def write_mp_label(self, egpr_iter_index):
        resolution = self.resolution
        num_labels = self.num_labels
        num_segs = self.num_segs
        files = self.out_files['mp']
        mp_label_filename = files['mp_label_filename'].format(egpr_iter = egpr_iter_index)
        with open(mp_label_filename, "wb") as f:
            # read posterior files from self.posterior_tmpls
            for window_index, (window_chrom, window_start, window_end, _, frame_offset) in enumerate(self.windows):
                # read segway posterior for this window
                window_num_frames = ceildiv(window_end - window_start, resolution)
                posteriors = [[10 for i in range(num_labels)] for j in range(window_num_frames)]
                for label_index in range(num_labels):
                    post_fname = self.out_files['post']['post_filename'].format(egpr_iter = egpr_iter_index,label_index=label_index, window_index=window_index)
                    with maybe_gzip_open(post_fname, "r") as post:
                        for line in post:
                            row = line.split()
                            chrom, start, end, prob = row[:4]
                            start = int(start)
                            end = int(end)
                            prob = float(prob) / 100
                            # The bed entry should be within the window
                            assert chrom == window_chrom
                            assert start >= window_start
                            assert end <= window_end
                            # segway's posteriors should line up with the resolution
                            if (((end - start) % resolution) != 0):
                                print(end, start, (end - start) % resolution, post_fname)
                            assert ((end - start) % resolution) == 0
                            assert ((start - window_start) % resolution) == 0
                            num_obs = ceildiv(end - start, resolution)
                            first_obs_index = int((start - window_start) / resolution)
                            for obs_index in range(first_obs_index, first_obs_index+num_obs):
                                posteriors[obs_index][label_index] = prob

                measure_label_fmt = "%sf" % num_segs
                for frame_index in range(window_num_frames):
                    # add psuedocounts to avoid breaking measure prop
                    posteriors[frame_index] = [((posteriors[frame_index][i] + 1e-20) /
                                            (1 + 1e-20*num_labels))
                                            for i in range(len(posteriors[frame_index]))]
                    # Ensure the posteriors sum to one (approximately)
                    assert (abs(sum(posteriors[frame_index]) - 1) < 0.01)   
                    # Write out the frame's posterior as a label for EGBR
                    f.write(struct.pack(measure_label_fmt, *posteriors[frame_index]))

    def write_mp_post(self, egpr_iter_index):
        files = self.out_files['mp']
        mp_graph_filename = files['mp_graph_filename']
        mp_label_filename = files['mp_label_filename'].format(egpr_iter = egpr_iter_index)

        
        mp_trans_filename = files['mp_trans_filename']
        mp_post_filename = files['mp_post_filename']
        mp_obj_filename = files['mp_obj_filename']

        cmd = ["measureProp/MP_large_scale",
            "-inputGraphName", mp_graph_filename,
            "-transductionFile", mp_trans_filename,
            "-labelFile", mp_label_filename,
            "-numThreads", "1",
            "-outPosteriorFile", mp_post_filename,
            "-numClasses", str(self.num_labels),
            "-mu", str(self.egpr_mu),
            "-nu", str(self.egpr_nu),
            "-selfWeight", str(self.egpr_alpha),
            "-nWinSize", "1",
            "-printAccuracy", "false",
            "-measureLabels", "true",
            "-maxIters", str(self.measure_prop_num_iters),
            "-outObjFile", mp_obj_filename,
            "-useSQL", "false"
            ]
        runCommand(cmd)
    
    def write_virtual_evidence(self):
        out_files = self.out_files['mp']
        mp_post_filename = out_files['mp_post_filename']
        out_files = self.out_files['ve']
        virtual_evidence_filename = out_files['virtual_evidence_filename']
        windows = self.windows
        node_frames = self.node_frames
        num_segs = self.num_segs
        num_frames = self.num_frames
        num_labels = self.num_labels
        mp_weight = self.mp_weight
        resolution = self.resolution
        with open(mp_post_filename, "rb") as mp_file, \
            open(virtual_evidence_filename, "w") as ve_file:
            header_fmt = "IH"
            frame_fmt = "%sf" % num_segs
            # Remove first line containing metadata
            num_nodes, num_classes = struct.unpack(header_fmt, mp_file.read(struct.calcsize(header_fmt)))
            assert (num_classes == num_segs)
            assert (num_nodes == num_frames)
            node_fmt = "I%sf" % num_segs
            for window_index, (window_chrom, window_start, window_end, _, frame_offset) \
                in enumerate(windows):
                window_posts = []
                window_num_frames = ceildiv(window_end-window_start, resolution)
                for i in range(window_num_frames):
                    # Read MP posteriors from f
                    line = struct.unpack(node_fmt,
                                        mp_file.read(struct.calcsize(node_fmt)))
                    index = line[0]
                    post = array(line[1:], dtype=float64)
                    # Transform posterior with mp_weight parameter, normalize and take log
                    post = power(post, mp_weight) + 1e-250
                    post = post / np.sum(post)
                    # Write out virtual_evidence 
                    start = window_start + resolution*i
                    end = window_start + resolution*(i+1)
                    for j in range(num_labels):
                        print(window_chrom, start, end, j , post[j], sep = "\t", file = ve_file)
    
    def write_annotation(self):
        postdir_path = self.out_files['post']['post_outdir_path']
        include_coords_filename = self.in_files['train']['include_coords_filename']
        #exclude_coords_filename = self.in_files['train']['exclude_coords_filename']
        genomedata_filename = self.in_files['train']['genomedata_filename']
        traindir_path = self.out_files['train']['traindir_path']

        virtual_evidence_filename = self.out_files['ve']['virtual_evidence_filename']
        cmd = ["segway", "posterior-init", \
        "--include-coords=" + include_coords_filename, \
        "--virtual-evidence=" + virtual_evidence_filename, \
        genomedata_filename, traindir_path, postdir_path]
        runCommand(cmd)

        cmd = ["segway", "posterior-run", "--virtual-evidence=" + virtual_evidence_filename, genomedata_filename, traindir_path, postdir_path]
        runCommand(cmd)

        cmd = ["segway", "posterior-finish", genomedata_filename, traindir_path, postdir_path]
        runCommand(cmd)

if __name__ == '__main__':
    egpr = EGPRInstance(biosample = biosample)
    egpr.neighbours = egpr.write_mp_graph()
    # with open(egpr.out_files['mp']['mp_graph_neighbours_filename']) as f:
    #     egpr.neighbours = ast.literal_eval((f.read()))

    for i in range(egpr.num_egpr_iters):
        egpr.write_segway_posteriors(i)
        egpr.write_mp_label(i)
        egpr.write_mp_post(i)
        egpr.write_virtual_evidence()
    
    egpr.write_annotation()




