from __future__ import print_function
from lib import models, graph, coarsening, utils, mesh_sampling
from lib.visualize_latent_space import LatentSpaceVisualization
from projections.projection import Projection
from projections.projection_ui import ProjectionUI
from projections.pca_projection import PCAProjection
import numpy as np
import json
import os
import copy
import argparse
from facemesh import FaceData
from opendr.topology import get_vert_connectivity
import time
import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('--name', default='bareteeth', help='facial_motion| lfw ')
    parser.add_argument('--data', default='data/bareteeth', help='facial_motion| lfw ')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train (default: 2)')
    parser.add_argument('--eval_frequency', type=int, default=200, help='eval frequency')
    parser.add_argument('--filter', default='chebyshev5', help='filter')
    parser.add_argument('--nz', type=int, default=8, help='Size of latent variable')
    parser.add_argument('--lr', type=float, default=8e-3, help='Learning Rate')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--viz', type=int, default=0, help='visualize while test')
    parser.add_argument('--loss', default='l1', help='l1 or l2')
    parser.add_argument('--mesh1', default='m1', help='for mesh interpolation')
    parser.add_argument('--mesh2', default='m1', help='for mesh interpolation')

    args = parser.parse_args()

    np.random.seed(args.seed)
    nz = args.nz
    print("Loading data .. ")
    reference_mesh_file = 'data/template.obj'
    facedata = FaceData(nVal=100, train_file=args.data+'/train.npy',
        test_file=args.data+'/test.npy', reference_mesh_file=reference_mesh_file, pca_n_comp=nz)

    ds_factors = [4,4,4,4]	# Sampling factor of the mesh at each stage of sampling
    print("Generating Transform Matrices ..")

    # Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
    # the mesh 4 times. Each time the mesh is sampled by a factor of 4

    M,A,D,U = mesh_sampling.generate_transform_matrices(facedata.reference_mesh, ds_factors)

    A = list(map(lambda x:x.astype('float32'), A))
    D = list(map(lambda x:x.astype('float32'), D))
    U = list(map(lambda x:x.astype('float32'), U))
    p = list(map(lambda x:x.shape[0], A))

    X_train = facedata.vertices_train.astype('float32')
    X_val = facedata.vertices_val.astype('float32')
    X_test = facedata.vertices_test.astype('float32')

    print("Computing Graph Laplacians ..")
    L = [graph.laplacian(a, normalized=True) for a in A]

    n_train = X_train.shape[0]
    params = dict()
    params['dir_name']       = args.name
    params['num_epochs']     = args.num_epochs
    params['batch_size']     = args.batch_size
    params['eval_frequency'] = args.eval_frequency
    # Building blocks.
    params['filter']         = args.filter
    params['brelu']          = 'b1relu'
    params['pool']           = 'poolwT'
    params['unpool']		 = 'poolwT'


    # Architecture.
    params['F_0']            = int(X_train.shape[2])  # Number of graph input features.
    params['F']              = [16, 16, 16, 32]  # Number of graph convolutional filters.
    params['K']              = [6, 6, 6, 6]  # Polynomial orders.
    params['p']              = p #[4, 4, 4, 4]    # Pooling sizes.
    params['nz']              = [nz]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['which_loss']     = args.loss
    params['nv']             = facedata.n_vertex
    params['regularization'] = 5e-4
    params['dropout']        = 1
    params['learning_rate']  = args.lr
    params['decay_rate']     = 0.99
    params['momentum']       = 0.9
    params['decay_steps']    = n_train / params['batch_size']

    model = models.coma(L=L, D=D, U=U, **params)

    if args.mode in ['test']:
        if not os.path.exists('results'):
            os.makedirs('results')
        predictions, loss = model.predict(X_test, X_test)
        print("L1 Loss= ", loss)
        euclidean_loss = np.mean(np.sqrt(np.sum((facedata.std*(predictions-facedata.vertices_test))**2, axis=2)))
        print("Euclidean loss= ", euclidean_loss)
        np.save('results/'+args.name+'_predictions', predictions)
        if args.viz:
            from psbody.mesh import MeshViewers
            viewer_recon = MeshViewers(window_width=800, window_height=1000, shape=[5, 4], titlebar='Mesh Reconstructions')
            for i in range(predictions.shape[0]//20):
                facedata.show_mesh(viewer=viewer_recon, mesh_vecs=predictions[i*20:(i+1)*20], figsize=(5,4))
                time.sleep(0.1)
    elif args.mode in ['sample']:
        meshes = facedata.get_normalized_meshes(args.mesh1, args.mesh2)
        features = model.encode(meshes)
    elif args.mode in ['latent']:
        LatentSpaceVisualization(model, facedata)
    elif args.mode in ['project']:
        proj_types = ('coma_mds', 'mds', 'tsne', 'pca_mds')
        projections = [Projection(proj_type, args.name, facedata, model) for proj_type in proj_types]
        inverses = ['rbf_coma', 'lamp', 'lamp', 'rbf']
        mesh_visualizer = LatentSpaceVisualization(model, facedata, viewer_size=(1080, 1080))
        ProjectionUI(args.name, projections, inverses, mesh_visualizer, fig_size=(14.0, 10.0), fig_pos=(1080, 0))
    elif args.mode in ['test_projection']:
        proj_types = ['pca_mds']
        projections = [Projection(proj_type, args.name, facedata, model, load_matrices=True)
                       for proj_type in proj_types]
        inverses = ['rbf']
        mesh_visualizer = LatentSpaceVisualization(model, facedata, viewer_size=(1080, 1080))
        ProjectionUI(args.name, projections, inverses, mesh_visualizer, fig_size=(14.0, 10.0), fig_pos=(1080, 0))
    elif args.mode in ['load_projection']:
        proj_types = ('coma_mds', 'mds', 'tsne', 'pca', 'pca_mds')
        projections = [Projection(proj_type, args.name, facedata, model, load_matrices=True)
                       for proj_type in proj_types]
        inverses = ['rbf_coma', 'lamp', 'lamp', 'pca', 'rbf']
        mesh_visualizer = LatentSpaceVisualization(model, facedata, viewer_size=(1080, 1080))
        ProjectionUI(args.name, projections, inverses, mesh_visualizer, fig_size=(14.0, 10.0), fig_pos=(1080, 0),
                     load_errors=True)
    elif args.mode in ['rbf_coma_test']:
        shape = facedata.vertices_test.shape
        facedata.vertices_test = np.resize(facedata.vertices_test, (10, shape[1], shape[2]))
        proj_types = ['coma_mds']
        projections = [Projection(proj_type, args.name + '_test', facedata, model, load_matrices=False)
                       for proj_type in proj_types]
        inverses = ['rbf_coma']
        mesh_visualizer = LatentSpaceVisualization(model, facedata, viewer_size=(1080, 1080))
        ProjectionUI(args.name, projections, inverses, mesh_visualizer, fig_size=(14.0, 10.0), fig_pos=(1080, 0))
    elif args.mode in ['pca_projection_test']:
        pca = PCAProjection(facedata.vertices_test, 2)
        proj = pca.proj_X
        reconstructed = np.array([pca.invert(proj[0]), pca.invert(proj[500]), pca.invert(proj[1000])])
        print('reconstructed shape: ' + str(reconstructed.shape))
        mesh_visualizer = LatentSpaceVisualization(model, facedata, viewer_size=(1080, 1080), figsize=(1, 3))
        mesh_visualizer.show(mesh=reconstructed)
        input()
    else:
        if not os.path.exists(os.path.join('checkpoints', args.name)):
            os.makedirs(os.path.join('checkpoints', args.name))
        with open(os.path.join('checkpoints', args.name +'params.json'),'w') as fp:
            saveparams = copy.deepcopy(params)
            saveparams['seed'] = args.seed
            json.dump(saveparams, fp)
        loss, t_step = model.fit(X_train, X_train, X_val, X_val)
