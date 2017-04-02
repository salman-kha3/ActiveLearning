/**
 * Created by root on 22.03.17.
 */
angular.module('myApp', [
  'ngRoute',
  'home',
  'signin'
]).
config(['$routeProvider', function($routeProvider) {
  $routeProvider.otherwise({redirectTo: '/home'});
}]);