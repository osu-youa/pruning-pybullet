import trimesh
import os
import numpy as np
import pickle


def convert_scene_to_trimesh(scene):
    if not isinstance(scene, trimesh.Scene):
        return scene
    return trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene.geometry.values()))


if __name__ == "__main__":

    MESH_POINT_DIST = 0.02
    POINTS_PER_TARGET = 150

    tree_dir = os.path.join('models', 'trees')
    files = [x for x in os.listdir(tree_dir) if x.endswith('.obj') and '-' not in x]
    for file in files:
        path = os.path.join(tree_dir, file)
        annotations = path.replace('.obj', '-annotations.obj')
        collisions = path.replace('.obj', '-collision.obj')
        output_file = path.replace('.obj', '.annotations')

        if os.path.exists(output_file):
            print('Annotations for {} loaded, skipping'.format(file))
            continue

        if not os.path.exists(annotations):
            print('[!] No annotations file found for {}, extract faces in Blender!'.format(file))
            continue

        if not os.path.exists(collisions):
            print('[!] No collision file found for {}, run V-HACD! (Set 5mil vox, gamma and concavity to 0'.format(file))
            continue

        base_mesh = convert_scene_to_trimesh(trimesh.load(collisions))
        assert base_mesh.is_watertight
        annotation_mesh = convert_scene_to_trimesh(trimesh.load(annotations))
        if len(annotation_mesh.faces) > 100:
            raise Exception('The annotation mesh for {} has too many faces! Did you forget to trim the original model?'.format(file))

        print('Processing annotations for {}'.format(file))
        collision_query = trimesh.proximity.ProximityQuery(base_mesh)

        metadata = {}
        annotated_locations = annotation_mesh.vertices[annotation_mesh.faces].mean(axis=1)
        for i, position in enumerate(annotated_locations):

            points_to_concat = []

            while len(points_to_concat) < POINTS_PER_TARGET:
                total_points = len(points_to_concat)
                remaining = POINTS_PER_TARGET - total_points

                # HACK to encourage closer points
                candidates_a = np.random.uniform(-MESH_POINT_DIST / 2, MESH_POINT_DIST / 2, size=(POINTS_PER_TARGET, 3)) + position
                candidates_b = np.random.uniform(-MESH_POINT_DIST, MESH_POINT_DIST, size=(POINTS_PER_TARGET, 3)) + position
                candidates = np.concatenate([candidates_a, candidates_b])
                contained_points = collision_query.signed_distance(candidates) > 0
                candidates = candidates[contained_points]

                to_add = min(remaining, len(candidates))
                points_to_concat.extend(candidates[:to_add])


            metadata[i] = {'position': position, 'points': np.array(points_to_concat)}


        with open(output_file, 'wb') as fh:
            pickle.dump(metadata, fh)
        print('Output annotations to: {}'.format(output_file))

