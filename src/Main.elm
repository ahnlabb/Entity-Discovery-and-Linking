module Main exposing (Document, Entity, Model(..), Msg(..), Page, body, documentDecoder, entityDecoder, errorString, getDocuments, getPrediction, getSel, init, localApi, main, newsApi, reduceEntities, reduceHelper, resultView, selectModel, spinner, strToOption, subscriptions, update, view)

import Browser
import Browser.Navigation as Navigation
import Dict
import Element exposing (Element, alignRight, alignTop, centerX, centerY, column, el, fill, height, none, padding, px, rgb255, row, spacing, text, width)
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Element.Input as Input
import Html exposing (option, select)
import Html.Attributes as HAttr exposing (class, style)
import Html.Events exposing (onInput)
import Http
import Json.Decode as Decode exposing (Decoder, dict, field, index, int, list, string)
import Json.Decode.Pipeline exposing (custom, required)
import Json.Encode as Encode
import Svg
import Svg.Attributes as SAttr
import Time
import Url exposing (Url)
import Url.Builder as UrlBuilder


main =
    Browser.application
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        , onUrlRequest = RequestedUrl
        , onUrlChange = ChangedUrl
        }



-- MODEL


type Model
    = Loading
    | Error Http.Error
    | NoModels
    | Done Page
    | News Page
    | LoadingNews


type alias Page =
    { selection : String
    , models : List String
    , reduceTags : Bool
    , prediction : Maybe Document
    , text : String
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


init : () -> Url -> Navigation.Key -> ( Model, Cmd Msg )
init _ { fragment } _ =
    case fragment of
        Just "news" ->
            ( LoadingNews, getDocuments )

        _ ->
            ( Loading, getDocuments )



-- UPDATE


type Msg
    = NewModels (Result Http.Error (List String))
    | NewNews (Result Http.Error String)
    | NewPrediction (Result Http.Error Document)
    | ToggleReduce Bool
    | NewSelection String
    | RequestedUrl Browser.UrlRequest
    | ChangedUrl Url


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model ) of
        ( NewModels result, Loading ) ->
            case result of
                Ok models ->
                    case models of
                        selection :: rest ->
                            ( Done (Page selection rest True Nothing ""), Cmd.none )

                        [] ->
                            ( NoModels, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewModels result, LoadingNews ) ->
            case result of
                Ok models ->
                    case models of
                        selection :: rest ->
                            ( Done (Page selection rest True Nothing ""), Cmd.none )

                        [] ->
                            ( NoModels, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewPrediction result, Done page ) ->
            case result of
                Ok prediction ->
                    ( Done { page | prediction = Just prediction }, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewPrediction result, News page ) ->
            case result of
                Ok prediction ->
                    ( News { page | prediction = Just prediction }, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( ToggleReduce bool, Done page ) ->
            ( Done { page | reduceTags = bool }, Cmd.none )

        ( NewNews result, News page ) ->
            case result of
                Ok article ->
                    ( News { page | text = article }
                    , getPrediction page.selection article
                    )

                Err e ->
                    ( Error e, Cmd.none )

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
            Browser.Document "page"
                [ Element.layout []
                    (body page)
                ]

        Loading ->
            Browser.Document "loading"
                [ Element.layout []
                    (el [] (text "loading"))
                ]

        Error e ->
            Browser.Document "error"
                [ Element.layout []
                    (el [] (text (errorString e)))
                ]

        NoModels ->
            Browser.Document "error"
                [ Element.layout []
                    (el [] (text "No models found"))
                ]

        News page ->
            Browser.Document "news"
                [ Element.layout []
                    (viewNews page)
                ]

        LoadingNews ->
            Browser.Document "loading"
                [ Element.layout []
                    (el [] (text "loading"))
                ]


body : Page -> Element Msg
body { models, selection, prediction, reduceTags } =
    column [ width fill, spacing 30 ]
        [ row [ width fill ]
            [ selectModel selection models
            , Input.checkbox []
                { onChange = ToggleReduce
                , icon = Input.defaultCheckbox
                , checked = reduceTags
                , label = Input.labelLeft [] (text "Reduce Tags")
                }
            ]
        , resultView selection prediction reduceTags
        ]


viewNews : Page -> Element Msg
viewNews { selection, prediction, reduceTags } =
    column [ width fill, spacing 30 ]
        [ row [ width fill ]
            [ Input.checkbox []
                { onChange = ToggleReduce
                , icon = Input.defaultCheckbox
                , checked = reduceTags
                , label = Input.labelLeft [] (text "Reduce Tags")
                }
            ]
        , resultView selection prediction reduceTags
        ]


selectModel : String -> List String -> Element Msg
selectModel selection models =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Element.html (select [ onInput NewSelection ] (selection :: models |> List.map strToOption)))


strToOption str =
    option [] [ Html.text str ]


getSel sel docs =
    let
        get dict key =
            Dict.get key dict
    in
    sel |> Maybe.andThen (get docs)


resultView : String -> Maybe Document -> Bool -> Element Msg
resultView selection prediction reduceTags =
    let
        pos x y tag attrs =
            tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

        textStyle sz =
            SAttr.style ("font-size: " ++ String.fromInt sz ++ "px;font-family: 'Source Code Pro', monospace;")

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
                    line origin start :: marked class start stop :: annotate stop string tail

                [] ->
                    []

        viewAnnotations =
            List.map Element.html >> Element.paragraph [ Font.family [ Font.typeface "Source Sans Pro", Font.sansSerif ] ]

        viewPrediction pred =
            case pred of
                Just document ->
                    reduceIfChecked document.entities |> annotate 0 document.text |> viewAnnotations

                Nothing ->
                    el [ width fill, alignTop ] spinner

        reduceIfChecked entities =
            if reduceTags then
                reduceEntities entities

            else
                entities
    in
    row [ width fill, spacing 50, padding 30 ]
        [ viewPrediction prediction
        ]


spinner =
    column [ centerX, alignTop ]
        [ Html.div [ class "lds-dual-ring" ] [] |> Element.html |> el [ centerX, padding 60 ]
        , el [ Font.center, width fill ] (Element.text "Predicting labels")
        ]


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
            "BadStatus: " ++ String.fromInt code

        Http.BadUrl str ->
            "BadUrl: " ++ str

        Http.NetworkError ->
            "NetworkError"

        Http.Timeout ->
            "Timeout"



-- HTTP


localApi : String
localApi =
    UrlBuilder.absolute [ "models" ] []


getDocuments : Cmd Msg
getDocuments =
    Http.get
        { url = localApi
        , expect = Http.expectJson NewModels (list string)
        }


getPrediction : String -> String -> Cmd Msg
getPrediction model text =
    Http.post
        { url = UrlBuilder.absolute [ "predict" ] [ UrlBuilder.string "model" model ]
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


newsApi : String -> String -> String -> String
newsApi lang query apiKey =
    UrlBuilder.crossOrigin "https://newsapi.org/"
        [ "v2", "everything" ]
        [ UrlBuilder.string "language" lang
        , UrlBuilder.string "q" query
        , UrlBuilder.string "apiKey" apiKey
        ]


getNews : String -> String -> String -> Cmd Msg
getNews lang query apiKey =
    Http.get
        { url = newsApi lang query apiKey
        , expect = Http.expectJson NewNews newsDecoder
        }


newsDecoder =
    field "articles" (index 0 (field "content" string))
