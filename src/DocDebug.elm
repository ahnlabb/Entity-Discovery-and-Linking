module Main exposing (..)

import Browser
import Http
import Url.Builder as Url
import Dict
import Element exposing (Element, el, text, row, column, alignRight, fill, width, height, rgb255, spacing, centerX, centerY, alignTop, padding, none, px, spacing)
import Element.Input as Input
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Html exposing (select, option)
import Html.Attributes as HAttr exposing (style, class)
import Html.Events exposing (onInput)
import Json.Decode as Decode exposing (Decoder, int, string, dict, list)
import Json.Decode.Pipeline exposing (required, custom)
import Json.Encode as Encode
import Time
import Svg
import Svg.Attributes as SAttr


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type Model
    = Loading
    | Error Http.Error
    | Done Page


type alias Page =
    { docs : Dict.Dict String Document
    , selection : Maybe String
    , reduceTags : Bool
    , prediction : Maybe Document
    }


type alias Document =
    { text : String
    , entities : List Entity
    }


type alias Entity =
    { start : Int
    , stop : Int
    , class : String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( Loading, getDocuments )



-- UPDATE


type Msg
    = NewDocuments (Result Http.Error (Dict.Dict String Document))
    | NewPrediction (Result Http.Error Document)
    | ToggleReduce Bool
    | NewSelection String


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model ) of
        ( NewDocuments result, Loading ) ->
            case result of
                Ok docs ->
                    ( Done (Page docs Nothing True Nothing), Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewPrediction result, Done page ) ->
            case result of
                Ok prediction ->
                    ( Done { page | prediction = Just prediction }, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewSelection string, Done page ) ->
            case getSel (Just string) page.docs of
                Just doc ->
                    ( Done { page | selection = Just string, prediction = Nothing }, getPrediction doc.text )

                Nothing ->
                    ( Done { page | selection = Just string }, Cmd.none )

        ( ToggleReduce bool, Done page ) ->
            ( Done { page | reduceTags = bool }, Cmd.none )

        ( _, _ ) ->
            ( model, Cmd.none )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions _ =
    Sub.none



-- VIEW


view model =
    case model of
        Done page ->
            Element.layout []
                (body page)

        Loading ->
            Element.layout []
                (el [] (text "loading"))

        Error e ->
            Element.layout []
                (el [] (text (errorString e)))


body : Page -> Element Msg
body { docs, selection, prediction, reduceTags } =
    column [ width fill, spacing 30 ]
        [ row [ width fill ]
            [ selectDoc docs
            , Input.checkbox []
                { onChange = ToggleReduce
                , icon = Input.defaultCheckbox
                , checked = reduceTags
                , label = Input.labelLeft [] (text "Reduce Tags")
                }
            ]
        , resultView docs selection prediction reduceTags
        ]


selectDoc : Dict.Dict String Document -> Element Msg
selectDoc dict =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Element.html (select [ onInput NewSelection ] (Dict.keys dict |> List.map strToOption)))


strToOption str =
    option [] [ Html.text str ]


getSel sel docs =
    let
        get dict key =
            Dict.get key dict
    in
        sel |> Maybe.andThen (get docs)


resultView : Dict.Dict String Document -> Maybe String -> Maybe Document -> Bool -> Element Msg
resultView docs selection prediction reduceTags =
    case getSel selection docs of
        Just doc ->
            let
                pos x y tag attrs =
                    tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

                textStyle sz =
                    SAttr.style ("font-size: " ++ (String.fromInt sz) ++ "px;font-family: 'Source Code Pro', monospace;")

                charWidth =
                    12

                charHeight =
                    15

                colorFromClass class =
                    case class of
                        "NAM-PER" ->
                            "#78CAD2"

                        "NAM-FAC" ->
                            "#63595C"

                        "NAM-LOC" ->
                            "#646881"

                        "NAM-ORG" ->
                            "#62BEC1"

                        "NAM-TTL" ->
                            "#5AD2F4"

                        "NAM-GPE" ->
                            "#72DDF7"

                        "NOM-PER" ->
                            "#F865B0"

                        "NOM-FAC" ->
                            "#E637BF"

                        "NOM-LOC" ->
                            "#FF928B"

                        "NOM-ORG" ->
                            "#FEC3A6"

                        "NOM-TTL" ->
                            "#FF3C38"

                        "NOM-GPE" ->
                            "#BB8588"

                        _ ->
                            "red"

                mark string =
                    let
                        length =
                            String.length string

                        w =
                            length * charWidth + 8 |> String.fromInt

                        height =
                            charHeight + 8

                        h =
                            height |> String.fromFloat

                        padding =
                            4
                    in
                        Svg.svg [ SAttr.width w, SAttr.height h, SAttr.viewBox ("0 0 " ++ w ++ " " ++ h) ]
                            [ Svg.g []
                                [ pos 0
                                    0
                                    Svg.rect
                                    [ SAttr.width w
                                    , SAttr.height h
                                    , SAttr.rx "5"
                                    , SAttr.ry "5"
                                    , SAttr.style ("fill:" ++ colorFromClass string ++ ";stroke:black;stroke-width:1;opacity:0.5")
                                    ]
                                    []
                                , pos padding (height - padding) Svg.text_ [ textStyle 20 ] [ Svg.text string ]
                                ]
                            ]

                annotate : Int -> String -> List Entity -> List (Html.Html Msg)
                annotate origin string ent =
                    let
                        annotation attrs marks begin end =
                            Html.span ([] ++ attrs)
                                ([ String.slice begin end string |> Html.text ] ++ marks)

                        line begin end =
                            Html.span [ style "line-height" "4.5em" ] [ plain begin end ]

                        plain begin end =
                            Html.text (String.slice begin end string)

                        marked class begin end =
                            Html.div
                                [ style "display" "inline-flex"
                                , style "flex-direction" "column"
                                , style "height" "3em"
                                ]
                                [ Html.div
                                    [ style "flex" "0 1 auto"
                                    , style "text-align" "center"
                                    , style "border" ("1px solid " ++ colorFromClass class)
                                    , style "border-radius" "5px"
                                    ]
                                    [ plain begin end ]
                                , Html.div
                                    [ style "flex" "0 1 auto"
                                    , style "text-align" "center"
                                    ]
                                    [ Html.div [] [ mark class ] ]
                                ]
                    in
                        case ent of
                            { start, stop, class } :: tail ->
                                (line origin start) :: (marked class start stop) :: (annotate stop string tail)

                            [] ->
                                [line origin (String.length string)]

                label =
                    List.map
                        (\str ->
                            el []
                                (Html.span [ colorFromClass str |> style "background-color" ] [ Html.text str ] |> Element.html)
                        )
                        [ "NAM-PER", "NAM-FAC", "NAM-LOC", "NAM-ORG", "NAM-TTL", "NAM-GPE", "NOM-PER", "NOM-FAC", "NOM-LOC", "NOM-ORG", "NOM-TTL", "NOM-GPE" ]

                viewAnnotations =
                    List.map Element.html >> Element.paragraph [ Font.family [ Font.typeface "Source Sans Pro", Font.sansSerif ] ]

                viewPrediction pred =
                    case pred of
                        Just document ->
                            reduceIfChecked document.entities |> annotate 0 document.text |> viewAnnotations

                        Nothing ->
                            el [ width fill, alignTop ]
                                (column [ centerX, alignTop ]
                                    [ Html.div [ class "lds-dual-ring" ] [] |> Element.html |> el [ centerX, padding 60 ]
                                    , el [ Font.center, width fill ] (Element.text "Predicting labels")
                                    ]
                                )

                reduceIfChecked entities =
                    if reduceTags then
                        reduceEntities entities
                    else
                        entities
            in
                row [ width fill, spacing 50, padding 30 ]
                    [ reduceIfChecked doc.entities |> annotate 0 doc.text |> viewAnnotations
                    , viewPrediction prediction
                    ]

        Nothing ->
            none


reduceHelper startPrev stopPrev classPrev entityList =
    case entityList of
        { start, stop, class } :: t ->
            let
                cur =
                    { start = start, stop = stop, class = class }

                className =
                    String.dropLeft 2 class
            in
                if classPrev /= className then
                    cur :: reduceEntities t
                else
                    case String.left 1 class of
                        "I" ->
                            reduceHelper startPrev stop className t

                        "E" ->
                            { start = startPrev, stop = stop, class = className } :: reduceEntities t

                        _ ->
                            { start = startPrev, stop = stopPrev, class = classPrev } :: reduceEntities (cur :: t)

        [] ->
            [ { start = startPrev, stop = stopPrev, class = classPrev } ]


reduceEntities entityList =
    case entityList of
        { start, stop, class } :: t ->
            case String.left 1 class of
                "B" ->
                    reduceHelper start stop (String.dropLeft 2 class) t

                "S" ->
                    { start = start, stop = stop, class = String.dropLeft 2 class } :: reduceEntities t

                "O" ->
                    reduceEntities t

                _ ->
                    { start = start, stop = stop, class = class } :: reduceEntities t

        [] ->
            []


errorString error =
    case error of
        Http.BadBody str ->
            "BadBody: " ++ str

        Http.BadStatus code ->
            "BadStatus: " ++ (String.fromInt code)

        Http.BadUrl str ->
            "BadUrl: " ++ str

        Http.NetworkError ->
            "NetworkError"

        Http.Timeout ->
            "Timeout"



-- HTTP


localApi : String
localApi =
    Url.absolute [ "gold" ] []


getDocuments : Cmd Msg
getDocuments =
    Http.get
        { url = localApi
        , expect = Http.expectJson NewDocuments (dict (documentDecoder "gold"))
        }


getPrediction : String -> Cmd Msg
getPrediction text =
    Http.post
        { url = Url.absolute [ "predict" ] []
        , body = Http.jsonBody (Encode.string text)
        , expect = Http.expectJson NewPrediction (documentDecoder "entities")
        }


documentDecoder name =
    Decode.succeed Document
        |> required "text" string
        |> required name (list entityDecoder)


entityDecoder =
    Decode.succeed Entity
        |> required "start" int
        |> required "stop" int
        |> required "class" string
