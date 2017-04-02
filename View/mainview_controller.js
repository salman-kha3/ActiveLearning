/**
 * Created by salman on 22.03.17.
 */
'use strict';
var app = angular.module('myApp', []);
app.controller('myCtrl', function($scope, $http) {

    $scope.sendQuery = function() {

        $http({
            method: 'POST',
            url: 'http://127.0.0.1:5000/sendQuery',
            data: $scope.review.rev

        }).then(function(response) {
            $scope.sentiment = response.data;
            console.log('mm');
        }, function(error) {
            console.log(error);
        });
    };
    $scope.updateClassifier = function() {
        $http({
            method: 'POST',
            url: 'http://127.0.0.1:5000/update',
            data: {
                // review : $scope.review.rev,
                // sent : $scope.review.selectedSentiment
                review : $scope.review
            }

        }).then(function(response) {
            $scope.sentiment = response.data;
            console.log('mm');
        }, function(error) {
            console.log(error);
        });
    }
    

});