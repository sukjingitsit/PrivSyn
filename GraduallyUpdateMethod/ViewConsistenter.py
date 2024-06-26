import copy
import numpy as np
from GraduallyUpdateMethod.View import View

class ViewConsistenter:
    class SubsetWithDependency:
        def __init__(self, attributes_set):
            self.attributes_set = attributes_set
            self.dependency = set()

    def __init__(self, views, num_categories, iterations = 10):
        self.views = views
        self.num_categories = num_categories
        self.iterations = iterations

    def compute_dependency(self):
        subsets_with_dependency = {}
        ret_subsets = {}

        for key, view in self.views.items():
            new_subset = self.SubsetWithDependency(view.attributes_set)
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                attributes_intersection = subset_value.attributes_set & view.attributes_set

                if attributes_intersection:
                    if tuple(attributes_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(attributes_intersection)
                        subsets_with_dependency[tuple(attributes_intersection)] = intersection_subset

                    if not tuple(attributes_intersection) == subset_key:
                        subsets_with_dependency[subset_key].dependency.add(tuple(attributes_intersection))
                    new_subset.dependency.add(tuple(attributes_intersection))

            subsets_with_dependency[tuple(view.attributes_set)] = new_subset

        for subset_key, subset_value in subsets_with_dependency.items():
            if len(subset_key) == 1:
                subset_value.dependency = set()

            ret_subsets[subset_key] = subset_value

        return subsets_with_dependency

    def consist_views(self):
        def find_subset_without_dependency():
            for key, subset in subsets_with_dependency_temp.items():
                if not subset.dependency:
                    return key, subset

            return None, None

        def find_views_containing_target(target):
            result = []

            for key, view in self.views.items():
                if target <= view.attributes_set:
                    result.append(view)

            return result

        def consist_on_subset(target):
            target_views = find_views_containing_target(target)

            common_view_indicator = np.zeros(self.num_categories.shape[0])
            for index in target:
                common_view_indicator[index] = 1

            common_view = View(common_view_indicator, self.num_categories)
            common_view.initialize_consist_parameters(len(target_views))

            for index, view in enumerate(target_views):
                common_view.project_from_bigger_view(view, index)

            common_view.calculate_delta()
            if np.sum(np.absolute(common_view.delta)) > 1e-3:
                for index, view in enumerate(target_views):
                    view.update_view(common_view, index)

        def remove_subset_from_dependency(target):
            for _, subset in subsets_with_dependency_temp.items():
                if tuple(target.attributes_set) in subset.dependency:
                    subset.dependency.remove(tuple(target.attributes_set))

        for key, view in self.views.items():
            view.calculate_tuple_key()
            view.generate_attributes_index_set()
            view.sum = np.sum(view.count)

        subsets_with_dependency = self.compute_dependency()
        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            consist_on_subset(set())

            for key, view in self.views.items():
                view.sum = np.sum(view.count)

            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)

            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()

                if not subset:
                    break

                consist_on_subset(subset.attributes_set)
                remove_subset_from_dependency(subset)
                subsets_with_dependency_temp.pop(key, None)

            nonneg_view_count = 0

            for key, view in self.views.items():
                if (view.count < 0.0).any():
                    view.non_negativity()
                    view.sum = np.sum(view.count)
                else:
                    nonneg_view_count += 1

                if nonneg_view_count == len(self.views):
                    non_negativity = False

            iterations += 1
            print(f"Iteration {iterations} of {self.iterations} completed to consist the marginal views")

        for key, view in self.views.items():
            view.sum = np.sum(view.count)
            view.normalize_count = view.count if view.sum <= 0 else view.count / view.sum

if __name__ == '__main__':
    pass